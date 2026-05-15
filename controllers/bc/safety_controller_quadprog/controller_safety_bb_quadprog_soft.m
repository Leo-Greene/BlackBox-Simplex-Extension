function [a, fit_val, exit_flag, prev_sol, history, a_h] = controller_safety_bb_quadprog_soft(pos, vel, params, opt)
%% controller_safety_bb_quadprog - Run safety-critical MPC controller using quadprog
%
% Input:
%   - pos          % 2 x n - 当前位置
%   - vel          % 2 x n - 当前速度
%   - params       % 参数结构体 (需要包含 tgt, n, h_bc, dt, a_max, wt, w_orient 等)
%   - opt          % (可选) 额外的求解器选项

    n = params.n;
    h_bc = params.h_bc;
    dim = 2; 
    N_vars = dim * n * h_bc; % 决策变量 U 的总维度

    % --- 0. 构造优化边界 (lb, ub) 和初值 (u_init) ---
    lb = -params.a_max * ones(N_vars, 1);
    ub =  params.a_max * ones(N_vars, 1);
    
    u_init = zeros(N_vars, 1); 

    % --- 1. 计算基础预测轨迹和响应矩阵 ---
    % dynamics：分成V和P两个部分更新
    [P_base, V_base, Bp, Bv] = compute_nominal_predictions(pos, vel, params);

    % =========================================================================
    % --- 1.5 新增：构造基于车辆间排斥力的期望速度 V_des (磁铁效应) ---
    % 目标：让车辆的期望速度主动远离附近的其他车辆，产生发散速度
    % =========================================================================
    
    % 将 P_base 和 V_base 重塑为 3D 矩阵 [dim, N, H] 方便按步和按车辆提取
    P_3d = reshape(P_base, 2, n, h_bc);
    V_3d = reshape(V_base, 2, n, h_bc);
    V_des_3d = zeros(2, n, h_bc);

    % 排斥力增益和作用范围（可调参数）
    k_repulse = 5.0; % 排斥力强度，越大越想散开
    dist_influence = 4.0; % 在多少距离内产生排斥力

    for k = 1:h_bc
        for i = 1:n
            p_i = P_3d(:, i, k);
            v_i = V_3d(:, i, k);
            
            % 1. 计算来自其他所有车的排斥速度向量
            v_repulse = [0; 0];
            for j = 1:n
                if i ~= j
                    p_j = P_3d(:, j, k);
                    v_diff = p_i - p_j; % 从 j 指向 i 的向量
                    dist = norm(v_diff);
                    
                    if dist < dist_influence && dist > 1e-3
                        % 模拟 1/d^2 势场，距离越近排斥力越大
                        repulse_dir = v_diff / dist;
                        v_repulse = v_repulse + repulse_dir * (1 / (dist^2));
                    end
                end
            end
            
            % 2. 结合原有的逃离原点策略 (防止全部聚集在中心)
            p_norm = norm(p_i);
            if p_norm > 1e-6
                v_escape_origin = (p_i / p_norm) * norm(v_i);
            else
                v_escape_origin = v_i;
            end
            
            % 3. 最终的期望速度 = 原点逃逸 + 强烈的小车间排斥
            v_des_vec = v_escape_origin + k_repulse * v_repulse;
            
            % 限制一下期望速度的大小，防止数值爆炸导致 QP 求解困难
            if norm(v_des_vec) > params.v_max * 1.5
                v_des_vec = (v_des_vec / norm(v_des_vec)) * (params.v_max * 1.5);
            end
            
            V_des_3d(:, i, k) = v_des_vec;
        end
    end
    
    % 将计算好的 V_des_3d 拉直回列向量
    V_des = V_des_3d(:);

    % =========================================================================
    % --- 2. 构造二次规划目标函数矩阵 (H 和 f) ---
    % =========================================================================

    % 2.2 控制能量惩罚 (Control Effort Penalty)
    H_u = (2 / h_bc) * eye(N_vars); 

    % 2.3 速度方向惩罚代价 (Orientation Cost) -> || V_pred - V_des ||^2
    % V_pred = Bv*U + V_base;  我们要最小化 || Bv*U + (V_base - V_des) ||^2
    E_v = V_base - V_des;
    H_orient = 2 * params.w_orient * (Bv' * Bv); 
    f_orient = 2 * params.w_orient * (Bv' * E_v);

    % 2.4 合并所有的 H 和 f
    H = H_u + H_orient;
    f = f_orient; 

    % 让 H 绝对对称（消除浮点数误差，防止 quadprog 报错）
    H = (H + H') / 2; 

    % --- 3. 构造线性不等式约束 A*U <= b ---
    % 限速约束
    [A_vel, b_vel] = compute_velocity_constraints(V_base, Bv, params);

    % PrSBC 安全约束
    % [A_sbc, b_sbc] = compute_PrSBC_constraints(pos, vel, params);

    % [A_col, b_col] = compute_collision_constraints_linear(P_base, Bp, params);
    [A_col, b_col] = compute_collision_constraints_cbf(P_base, Bp, pos, params);

    % --- 在构造 A_ineq 之前 ---
    [n_constraints, ~] = size(A_col); % 获取防碰撞约束的数量

    % 1. 扩展决策变量：X = [U; epsilon]
    % epsilon 的维度等于 n_constraints
    N_vars_total = N_vars + n_constraints;

    % 2. 构造新的目标函数 H_new, f_new
    rho = 1e8; % 惩罚系数，建议取 1e5 ~ 1e8 之间
    H_new = blkdiag(H, 2 * rho * eye(n_constraints)); 
    f_new = [f; zeros(n_constraints, 1)];

    % 3. 构造新的不等式约束 [A_col, -I] * [U; eps] <= b_col
    % 注意：限速约束 A_vel 通常保持为硬约束，不需要松弛
    A_col_soft = [A_col, -eye(n_constraints)];
    A_vel_pad  = [A_vel, zeros(size(A_vel, 1), n_constraints)];

    A_ineq_new = [A_col_soft; A_vel_pad];
    b_ineq_new = [b_col; b_vel];

    % 4. 设置边界约束
    % U 的边界保持不变，epsilon 必须 >= 0
    lb_new = [lb; zeros(n_constraints, 1)];
    ub_new = [ub; inf * ones(n_constraints, 1)];

    % idx_end = (params.h_bc - 1) * dim * n + 1 : params.h_bc * dim * n;

    % Aeq_term = Bv(idx_end, :);
    % beq_term = -V_base(idx_end);

    % Aeq_term_pad = [Aeq_term, zeros(length(idx_end), n_constraints)];

    % 调用 quadprog 时，传入 Aeq 和 beq

    % 5. 调用求解器
    if ~isempty(opt)
        qp_opt = opt;
    else
        qp_opt = optimoptions('quadprog', 'Display', 'off', 'MaxIterations', 8000);
    end
    u_init_new = [u_init; zeros(n_constraints, 1)];
    [X_opt, fit_val, exit_flag] = quadprog(H_new, f_new, A_ineq_new, b_ineq_new, [], [], lb_new, ub_new, u_init_new, qp_opt);

    % 6. 提取 U
    U_opt = X_opt(1:N_vars);
    
    % --- 5. 处理结果与提取指令 ---
    if exit_flag < 0
        % 发生死锁/无解时，使用紧急分离策略
        warning('[BC] QP Solver failed with exit_flag = %d. Using emergency separation strategy.', exit_flag);
        a = zeros(2, n);
        for i = 1:n
            % 求其他所有车对车 i 的排斥力方向
            repulse_dir = [0; 0];
            for j = 1:n
                if i ~= j
                    v_diff = pos(:, i) - pos(:, j);
                    if norm(v_diff) < 3.0 % 距离近的车产生排斥
                        repulse_dir = repulse_dir + v_diff / (norm(v_diff)^3 + 1e-3);
                    end
                end
            end
            if norm(repulse_dir) > 1e-4
                % 沿排斥方向给予最大加速度
                a(:, i) = (repulse_dir / norm(repulse_dir)) * params.a_max;
            end
        end
        prev_sol = zeros(N_vars, 1);
        a_h = zeros(n, 2, h_bc);
        % 把这个紧急避让指令复制给整个时域
        for k = 1:h_bc
            a_h(:,:,k) = a'; 
        end
    else
        u_step_1_vec = U_opt(1 : 2*n);
        a = reshape(u_step_1_vec, 2, n); 
        prev_sol = U_opt; 
        a_h = u2acc(U_opt, n, h_bc);
    end

    history=[];
    

end