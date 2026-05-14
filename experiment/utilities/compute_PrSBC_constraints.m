%% compute_PrSBC_constraints
% Computes the constraints for a probabilistic Control Barrier Function (PrSBC) for a group of agents.

function [A_sbc, b_sbc] = compute_PrSBC_constraints(pos, vel, params)
    % pos: 2 x n 当前位置
    % vel: 2 x n 当前速度

    n = params.n;
    h_ac = params.h_ac;
    dim = 2;
    N_vars = dim * n * h_ac; % U 的总长度

    % --- 1. 离散动力学系统矩阵 (半隐式欧拉) ---
    dt = params.dt;
    F_sys = [ 1, 0, dt, 0;  
              0, 1, 0, dt;  
              0, 0, 1, 0;   
              0, 0, 0, 1 ]; 
              
    G_sys = [ dt^2,   0;    
              0,    dt^2;   
              dt,     0;    
              0,     dt  ]; 
    
    % 提取 G 矩阵关于位置的前两行
    G_p = G_sys(1:2, :); 

    % --- 2. 初始化 QP 约束矩阵 ---
    num_constraints = nchoosek(n, 2);
    A_sbc = zeros(num_constraints, N_vars);
    b_sbc = zeros(num_constraints, 1);
    
    % --- 3. 读取控制参数 (提供默认值以防报错) ---
    if isfield(params, 'gamma'), gamma = params.gamma; else, gamma = 0.5; end
    if isfield(params, 'confidence'), Confidence = params.confidence; else, Confidence = 0.99; end
    
    % 提取三种噪声的标准差 (sigma)
    if isfield(params, 'sigma_obs_pos'), sigma_obs_p = params.sigma_obs_pos; else, sigma_obs_p = 0.02; end
    if isfield(params, 'sigma_obs_vel'), sigma_obs_v = params.sigma_obs_vel; else, sigma_obs_v = 0.02; end
    if isfield(params, 'sigma_proc'),    sigma_proc = params.sigma_proc; else, sigma_proc = 0.01; end

    count = 1;
    for i = 1:(n-1)
        for j = (i+1):n
            % =========================================================
            % 步骤 A: 提取当前状态, 预测未来名义状态
            % =========================================================
            dp_curr = pos(:, i) - pos(:, j);
            dv_curr = vel(:, i) - vel(:, j);
            
            % 拼成 4x1 相对状态向量
            X_rel_curr = [dp_curr; dv_curr]; 
            
            % 乘转移矩阵得到下一时刻的相对状态
            X_rel_next_base = F_sys * X_rel_curr; 
            
            % 提取位置部分 \hat{p}_{t+1}
            p_next_base = X_rel_next_base(1:2); 
            norm_p_next = norm(p_next_base);

            % =========================================================
            % 步骤 B: 计算当前时刻的严格安全裕度 h_t
            % =========================================================
            % 单车方差是 sigma^2，相对(差值)方差是 2*sigma^2，相对标准差是 sqrt(2)*sigma
            sigma_rel_p = sqrt(2) * sigma_obs_p; 
            
            % 调用 2D 逆累积函数
            dist_worst = get_2d_probabilistic_distance(dp_curr, sigma_rel_p, Confidence);
            
            % 使用最保守的距离算 h_t
            h_t = dist_worst^2 - params.R_safe^2;

            % =========================================================
            % 步骤 C: 计算未来的不确定性膨胀 (总噪声传播)
            % =========================================================
            % 两车相对综合总方差 = 位置观测方差 + 速度导致的位移方差 + 过程执行方差
            % (注意每项都要乘 2，因为是两辆车互相独立的误差累加)
            sigma_total_sq = 2 * sigma_obs_p^2 + 2 * (sigma_obs_v * dt)^2 + 2 * sigma_proc^2;
            sigma_total = sqrt(sigma_total_sq);

            % 瑞利分布逆函数：计算出最坏情况下的概率边界半径
            epsilon_total = sigma_total * sqrt(-2 * log(1 - Confidence));

            % 构造柯西-施瓦茨噪声扣减项
            noise_penalty = 2 * norm_p_next * epsilon_total;

            % =========================================================
            % 步骤 D: 构造 A 矩阵和 b 向量
            % =========================================================
            idx_i = (2*i - 1) : (2*i);
            idx_j = (2*j - 1) : (2*j);

            % A 矩阵系数：利用名义预测方向投影
            a_coeff = -2 * p_next_base' * G_p; 
            
            A_sbc(count, idx_i) = a_coeff;
            A_sbc(count, idx_j) = -a_coeff; 

            % b 向量：减去保守距离和未来惩罚项
            b_val = norm_p_next^2 - params.R_safe^2 - (1 - gamma) * h_t - noise_penalty;
            b_sbc(count) = b_val;
            
            count = count + 1;
        end
    end
end

% =========================================================
% 附属子函数：2D 概率边界计算
% =========================================================
function dp_worst_norm = get_2d_probabilistic_distance(dp_measured, sigma_rel, confidence)
    if confidence >= 1.0
        epsilon_obs = 3 * sigma_rel; 
    else
        epsilon_obs = sigma_rel * sqrt(-2 * log(1 - confidence));
    end
    measured_norm = norm(dp_measured);
    if measured_norm <= epsilon_obs
        dp_worst_norm = 0; 
    else
        dp_worst_norm = measured_norm - epsilon_obs;
    end
end