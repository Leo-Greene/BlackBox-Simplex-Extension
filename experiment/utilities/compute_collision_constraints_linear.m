function [A_col, b_col] = compute_collision_constraints_linear(P_base, Bp, params)
% 基于预测轨迹一阶泰勒展开的多步线性防撞约束
% P_base: 惯性滑行预测位置 (2*N*H x 1)
% Bp: 位置响应矩阵 (2*N*H x 2*N*H)

    N = params.n;
    H = params.h_bc;
    dim = 2;
    N_vars = dim * N * H;
    
    % QP中的安全距离最好比DM中的dmin稍微大一点点（例如DM是1.7，这里设1.75），作为缓冲
    R_safe = 1.75; 

    num_pairs = N * (N - 1) / 2;
    % 每对智能体在每个预测步都产生1个约束
    A_col = zeros(num_pairs * H, N_vars);
    b_col = zeros(num_pairs * H, 1);

    row = 1;
    for k = 1:H % 遍历预测步
        for i = 1:N-1 % 遍历智能体 i
            for j = i+1:N % 遍历智能体 j
                
                % 获取第 k 步时，智能体 i 和 j 在 P_base 中的索引
                idx_i = (k-1)*dim*N + (i-1)*dim + (1:2);
                idx_j = (k-1)*dim*N + (j-1)*dim + (1:2);

                % 提取名义预测位置
                pi_base = P_base(idx_i);
                pj_base = P_base(idx_j);

                % 计算相对向量和距离
                v_ij = pi_base - pj_base;
                dist = norm(v_ij);

                % 计算单位法向量 n_ij (从 j 指向 i)
                if dist < 1e-4
                    n_ij = [1; 0]; % 防止重合时的除零错误
                else
                    n_ij = v_ij / dist;
                end

                % 提取响应矩阵中对应的行
                Bp_i = Bp(idx_i, :);
                Bp_j = Bp(idx_j, :);

                % 构造约束：-n_ij^T * (Bp_i - Bp_j) * U <= dist - R_safe
                A_col(row, :) = -n_ij' * (Bp_i - Bp_j);
                b_col(row) = dist - R_safe;

                row = row + 1;
            end
        end
    end
end