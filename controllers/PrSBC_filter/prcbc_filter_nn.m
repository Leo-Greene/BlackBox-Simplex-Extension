function [a, fit_val, exit_flag, prev_sol, history, a_h] = prcbc_filter_nn(pos, vel, params, opt)
%% prcbc_filter_nn - 作为底层的 PrSBC 安全过滤器 (使用神经网络修正)
% 
% 核心逻辑:
%   接收上层 Simplex(DM) 传来的名义指令 u_cmd，
%   在只预测 1 步的 PrSBC 约束下，利用 QP 找到离 u_cmd 最近的绝对安全指令。
%
% Input:
%   - pos          % 2 x n - 当前位置
%   - vel          % 2 x n - 当前速度
%   - params       % 必须包含 a_max, n, dt 等，且需要包含上层传来的参考指令 params.u_cmd
%   - opt          % (可选) 额外的求解器选项

    n = params.n;
    dim = 2; 
    
    % =====================================================================
    % 1. 强制预测步长为 1 (舵手只看眼前一步)
    % =====================================================================
    % 为了防止复用外部 AC/BC 的长预测域，这里构建一个专属的 filter 参数结构体
    params_filter = params;
    params_filter.h_ac = 1; 
    
    N_vars = dim * n * 1; % 决策变量 U 的总维度 (仅1步)

    % =====================================================================
    % 2. 获取上层(船长)下达的参考指令 u_cmd
    % =====================================================================
    if isfield(params, 'u_cmd')
        u_cmd = params.u_cmd; % 尺寸应为 2 x n
    else
        % 如果上层没有传，兜底设为 0 (刹车)
        warning('params.u_cmd is missing! Fallback to zero command.');
        u_cmd = zeros(dim, n); 
    end
    
    % 将 2 x n 的矩阵展平为 2n x 1 的列向量，作为 QP 的追踪目标
    u_ref = u_cmd(:); 

    % =====================================================================
    % 3. 构造 QP 的目标函数矩阵 (H 和 f)
    % =====================================================================
    % 目标: min || U - u_ref ||^2 
    % 展开后: U'*U - 2*u_ref'*U + const
    % 对应 quadprog 标准形式: min 0.5 * U' * H * U + f' * U
    % 因此 H = 2*I, f = -2*u_ref
    
    H = 2 * eye(N_vars);
    f = -2 * u_ref;

    % =====================================================================
    % 4. 计算 PrSBC 避碰约束 (A * U <= b)
    % =====================================================================
    [A_sbc, b_sbc] = compute_PrSBC_constraints_nn(pos, vel, params_filter);

    % =====================================================================
    % 5. 构建边界约束 (lb, ub) 与初值
    % =====================================================================
    lb = -params.a_max * ones(N_vars, 1);
    ub =  params.a_max * ones(N_vars, 1);
    
    u_init = u_ref; % 将上层指令作为 QP 的初始猜测点(Warm Start)

    % =====================================================================
    % 6. 引入松弛变量 (Soft Constraints) 保证求解器永不崩溃
    % =====================================================================
    [n_cons, ~] = size(A_sbc); % 获取约束行数
    rho_sbc = 1e6; % 极大的软约束惩罚权重 (非万不得已绝不违反安全距离)
    
    % 扩展 H 和 f (加入 epsilon 的惩罚)
    H_new = blkdiag(H, 2 * rho_sbc * eye(n_cons)); 
    f_new = [f; zeros(n_cons, 1)];
    
    % 扩展不等式 A_sbc * U - epsilon <= b_sbc
    A_ineq_new = [A_sbc, -eye(n_cons)];
    b_ineq_new = b_sbc;
    
    % 扩展边界 (epsilon >= 0)
    lb_new = [lb; zeros(n_cons, 1)];
    ub_new = [ub; inf * ones(n_cons, 1)];
    
    % 扩展初值
    u_init_new = [u_init; zeros(n_cons, 1)];

    % =====================================================================
    % 7. 调用 QP 求解器
    % =====================================================================
    if nargin < 4 || isempty(opt)
        qp_opt = optimoptions('quadprog', 'Display', 'off', 'MaxIterations', 8000);
    else
        qp_opt = opt;
    end
        
    [X_opt, fit_val, exit_flag] = quadprog(H_new, f_new, A_ineq_new, b_ineq_new, ...
                                           [], [], lb_new, ub_new, u_init_new, qp_opt);

    % =====================================================================
    % 8. 处理结果与提取指令
    % =====================================================================
    if exit_flag < 0
        % 理论上加了软约束极少发生无解，除非输入数据包含 NaN 或 Inf
        warning('[PrSBC Filter] QP Solver failed with exit_flag = %d. Using emergency stop.', exit_flag);
        a = zeros(dim, n);
        fit_val = 1e6;   
        prev_sol = u_init; 
    else
        % 提取真实控制量 U (丢弃尾部的 epsilon 松弛变量)
        U_opt = X_opt(1:N_vars); 
        
        % 还原成 2 x n 的控制矩阵
        a = reshape(U_opt, dim, n); 
        prev_sol = U_opt; 
        
        if isfield(params, 'verbose') && params.verbose
            fprintf('[prcbc_filter_nn Verbose] QP Exit Flag: %d\n', exit_flag);
            diff_norm = norm(U_opt - u_ref);
            if diff_norm > 1e-3
                fprintf('  - Deviation from nominal command u_cmd (L2 Norm): %.4f\n', diff_norm);
                u_diff = reshape(U_opt - u_ref, dim, n);
                for i = 1:n
                    mod_norm = norm(u_diff(:, i));
                    if mod_norm > 0.05
                        fprintf('  - Agent %d modified significantly: [%.4f; %.4f] -> [%.4f; %.4f] (diff: %.4f)\n', ...
                            i, u_cmd(1,i), u_cmd(2,i), a(1,i), a(2,i), mod_norm);
                    end
                end
            else
                fprintf('  - Safety filter did NOT alter the nominal commands (fully safe).\n');
            end
        end
    end
    
    % 返回结构补齐 (因为只预测1步，历史轨迹直接为空，完整序列也是当前解)
    history = [];
    a_h = a; 

end