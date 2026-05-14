function [a, fit_val, exit_flag, prev_sol, history] = controller_cmpc_2d_quadprog_soft(pos, vel, params, opt)
%% controller_cmpc_2d - Run MPC controller using quadprog
%
% Input:
%   - pos          % 2 x n - 当前位置
%   - vel          % 2 x n - 当前速度
%   - params       % 参数结构体 (需要包含 tgt, n, h_ac, dt, a_max 等)
%   - opt          % (可选) 额外的求解器选项

    n = params.n;
    h_ac = params.h_ac;
    dim = 2; 
    N_vars = dim * n * h_ac; % 决策变量 U 的总维度

    % --- 0. 构造优化边界 (lb, ub) 和初值 (u_init) ---
    lb = -params.a_max * ones(N_vars, 1);
    ub =  params.a_max * ones(N_vars, 1);
    
    u_init = zeros(N_vars, 1); 
    
    tgt_vec = params.tgt(:); 
    P_tgt_expanded = repmat(tgt_vec, h_ac, 1);

    % --- 1. 计算基础预测轨迹和响应矩阵 ---
    % dynamics：分成V和P两个部分更新（目标函数只关心 P ， 限速约束只关心 V ， PrSBC 约束同时关心 P 和 V）
    [P_base, V_base, Bp, Bv] = compute_nominal_predictions(pos, vel, params);

    % --- 2. 构造二次规划目标函数矩阵 (H 和 f) ---
    E = P_base - P_tgt_expanded; 
    H_tgt = 2 * params.wt * (Bp' * Bp);
    f_tgt = 2 * params.wt * (Bp' * E);

    % 控制能量惩罚 (防止加速度突变震荡)
    H_u = (2 / h_ac) * eye(N_vars); 

    H = H_tgt + H_u;
    f = f_tgt; 

    % 让 H 绝对对称（消除浮点数误差，防止 quadprog 报错）
    H = (H + H') / 2; 

    % --- 3. 构造线性不等式约束 A*U <= b ---
    % 限速约束
    [A_vel, b_vel] = compute_velocity_constraints(V_base, Bv, params);

    % 防碰撞约束
    % [A_col, b_col] = compute_collision_constraints_linear(P_base, Bp, params);
    [A_col, b_col] = compute_collision_constraints_cbf(P_base, Bp, pos, params);

    % =========================================================================
    % --- 3.5 引入松弛变量 (Soft Constraints) ---
    % =========================================================================
    [n_col_constraints, ~] = size(A_col); % 获取防碰撞约束的数量
    
    % 软约束惩罚权重 (可适当调小一点，让 AC 在关键时刻更容易放弃硬抗)
    rho_ac = 1e6; 
    
    % 扩展 H 和 f 矩阵
    H_new = blkdiag(H, 2 * rho_ac * eye(n_col_constraints)); 
    f_new = [f; zeros(n_col_constraints, 1)];

    
    % 扩展不等式矩阵: 碰撞约束引入 -epsilon，限速约束不引入(补零)
    A_col_soft = [A_col, -eye(n_col_constraints)];
    A_vel_pad  = [A_vel, zeros(size(A_vel, 1), n_col_constraints)];
    
    A_ineq_new = [A_col_soft; A_vel_pad];
    b_ineq_new = [b_col; b_vel];
    
    % 扩展边界条件 (epsilon >= 0)
    lb_new = [lb; zeros(n_col_constraints, 1)];
    ub_new = [ub; inf * ones(n_col_constraints, 1)];
    


    % 扩展初值
    u_init_new = [u_init; zeros(n_col_constraints, 1)];

    % --- 4. 调用 QP 求解器 ---
    if ~isempty(opt)
        qp_opt = opt;
    else
        qp_opt = optimoptions('quadprog', 'Display', 'off', 'MaxIterations', 8000);
    end
        
    % 注意：传入的是扩展后的 _new 矩阵
    [X_opt, fit_val, exit_flag] = quadprog(H_new, f_new, A_ineq_new, b_ineq_new, [], [], lb_new, ub_new, u_init_new, qp_opt);

    % --- 5. 处理结果与提取指令 ---
    if exit_flag < 0
        % 求解失败时的安全容错处理（Fallback）
        warning('[AC] QP Solver failed with exit_flag = %d. Using safe fallback.', exit_flag);
        a = zeros(2, n); % 紧急刹车或保持上次指令
        fit_val = 1e6;   % 赋予一个惩罚值，避免空数组导致的崩溃
        prev_sol = u_init; 
    else
        % 正常求解，提取出属于控制指令 U 的部分 (丢弃 epsilon)
        U_opt = X_opt(1:N_vars); 
        
        % 提取第一步的控制指令
        u_step_1_vec = U_opt(1 : 2*n);
        a = reshape(u_step_1_vec, 2, n); % 完美还原成 2 x n
        prev_sol = U_opt; % 保存当前解供下一步 warm start 使用
    end
    
    % 返回空的 history
    history = [];
end