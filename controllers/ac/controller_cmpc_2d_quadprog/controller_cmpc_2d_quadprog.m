function [a, fit_val, exit_flag, prev_sol, history] = controller_cmpc_2d_quadprog(pos, vel, params, opt)
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
        % 求解失败时的安全容错处理（Fallback）
        warning('QP Solver failed with exit_flag = %d. Using safe fallback.', exit_flag);
        a = zeros(2, n); % 紧急刹车或保持上次指令
        prev_sol = u_init; 
    else
        % 正常求解，提取第一步的控制指令
        u_step_1_vec = U_opt(1 : 2*n);
        a = reshape(u_step_1_vec, 2, n); % 完美还原成 2 x n
        prev_sol = U_opt; % 保存当前解供下一步 warm start 使用
    end
    
    % 返回空的 history（如果不需要记录迭代历史的话）
    history = [];
end
