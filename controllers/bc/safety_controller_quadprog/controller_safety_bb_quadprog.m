function [a, fit_val, exit_flag, prev_sol, history, a_h] = controller_safety_bb_quadprog(pos, vel, params, opt)
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
    % --- 1.5 新增：构造方向对齐的期望速度 V_des (方案B核心) ---
    % 目标：让速度方向指向位置方向
    % =========================================================================
    V_des = zeros(size(V_base));
    P_base_2d = reshape(P_base, 2, []); % 转换为 2 x (n*h_bc) 以便按列提取向量
    V_base_2d = reshape(V_base, 2, []); 

    for i = 1:size(P_base_2d, 2)
        p = P_base_2d(:, i);
        v = V_base_2d(:, i);
        p_norm = norm(p);
        v_norm = norm(v);
        
        if p_norm > 1e-6
            % 期望速度方向 = P的方向，期望速度大小 = 基础预测速度的大小
            v_des_vec = (p / p_norm) * v_norm;
        else
            % 如果位置在原点，无明确方向，期望速度即为基础速度（不产生惩罚）
            v_des_vec = v; 
        end
        % 填入 V_des 列向量中
        V_des(2*i-1 : 2*i) = v_des_vec;
    end

    % =========================================================================
    % --- 2. 构造二次规划目标函数矩阵 (H 和 f) ---
    % =========================================================================

    % 2.2 控制能量惩罚 (Control Effort Penalty)
    H_u = (2 / h_bc) * eye(N_vars); 

    % 2.3 速度方向惩罚代价 (Orientation Cost) -> || V_pred - V_des ||^2
    % V_pred = Bv*U + V_base;  我们要最小化 || Bv*U + (V_base - V_des) ||^2
    E_v = V_base - V_des;
    % 假设 params.w_orient 是你在外部定义的权重
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
    [A_sbc, b_sbc] = compute_PrSBC_constraints(pos, vel, params);

    % 组合约束
    A_ineq = [A_sbc; A_vel];
    b_ineq = [b_sbc; b_vel];

    % --- 4. 调用 QP 求解器 ---
    if ~isempty(opt)
        qp_opt = opt;
    else
        qp_opt = optimoptions('quadprog', 'Display', 'off', 'MaxIterations', 8000);
    end
    [U_opt, fit_val, exit_flag] = quadprog(H, f, A_ineq, b_ineq, [], [], lb, ub, u_init, qp_opt);

    % --- 5. 处理结果与提取指令 ---
    if exit_flag < 0
        warning('QP Solver failed with exit_flag = %d. Using safe fallback.', exit_flag);
        a = zeros(2, n); 
        prev_sol = u_init; 
    else
        u_step_1_vec = U_opt(1 : 2*n);
        a = reshape(u_step_1_vec, 2, n); 
        prev_sol = U_opt; 
        a_h = zeros(n, 2, h_bc+1);
        a_h(:,:,1:end-1) = u2acc(U_opt, n, h_bc);
    end

    history=[];
    

end