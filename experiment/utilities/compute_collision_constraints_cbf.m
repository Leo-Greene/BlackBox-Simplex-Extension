function [A_col, b_col] = compute_collision_constraints_cbf(P_base, Bp, pos, params)
% compute_collision_constraints_cbf - 基于离散时间控制屏障函数 (DT-CBF) 的防撞约束
%
% Input:
%   P_base: 惯性滑行预测位置 (2*N*H x 1)
%   Bp:     位置响应矩阵 (2*N*H x N_vars)
%   pos:    当前真实位置 (2 x N)，用于计算 CBF 的初始状态 h(x_0)
%   params: 参数结构体
%
% Output:
%   A_col, b_col: 线性化后的不等式约束 A_col * U <= b_col

    N = params.n;
    dim = 2;
    
    % 动态推断预测时域 H，这样能同时兼容 AC 和 BC 不同的 horizon 长度
    H = length(P_base) / (dim * N); 
    N_vars = dim * N * H;
    
    % ==========================================
    % CBF 核心调参区域
    % ==========================================
    R_safe = 1.75; % 安全距离缓冲
    
    % gamma: CBF 衰减系数，取值范围 (0, 1]
    % - gamma 越小 (如 0.1)：系统越早开始避让，动作越平滑，倾向于擦肩绕行。
    % - gamma 越大 (如 0.8)：系统允许逼近到极限才猛打方向，倾向于紧急刹车推开。
    gamma = 0.05;   
    
    num_pairs = N * (N - 1) / 2;
    % 每对智能体在每个预测步都产生1个约束
    A_col = zeros(num_pairs * H, N_vars);
    b_col = zeros(num_pairs * H, 1);

    row = 1;
    for k = 1:H % 遍历预测步
        
        % 计算当前步的衰减权重: (1 - gamma)^k
        % 随着预测步数 k 增加，允许 h_k 的安全下界指数递减，这是实现“侧向绕行”的关键
        decay_factor = (1 - gamma)^k; 

        for i = 1:N-1 % 遍历智能体 i
            for j = i+1:N % 遍历智能体 j
                
                % 1. 计算当前时刻 (t=0) 的真实安全距离 h_current (即 h0)
                p_i_current = pos(:, i);
                p_j_current = pos(:, j);
                dist_current = norm(p_i_current - p_j_current);
                h_current = dist_current - R_safe;
                
                % 2. 获取第 k 步时，智能体 i 和 j 在 P_base 中的索引
                idx_i = (k-1)*dim*N + (i-1)*dim + (1:2);
                idx_j = (k-1)*dim*N + (j-1)*dim + (1:2);

                % 3. 提取名义预测位置
                pi_base = P_base(idx_i);
                pj_base = P_base(idx_j);

                % 计算名义相对向量和距离
                v_ij = pi_base - pj_base;
                dist_base = norm(v_ij);

                % 计算单位法向量 n_ij (梯度方向)
                if dist_base < 1e-4
                    n_ij = [1; 0]; % 防止重合时的除零错误
                else
                    n_ij = v_ij / dist_base;
                end

                % 4. 提取响应矩阵中对应的行
                Bp_i = Bp(idx_i, :);
                Bp_j = Bp(idx_j, :);

                % 5. 构造 CBF 线性约束
                % 理论公式：h_base_k + nabla_h * U >= decay_factor * h_current
                % 移项变号为 A*U <= b 的形式：
                A_col(row, :) = -n_ij' * (Bp_i - Bp_j);
                b_col(row) = (dist_base - R_safe) - decay_factor * h_current;

                row = row + 1;
            end
        end
    end
end