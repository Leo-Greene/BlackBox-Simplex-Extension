function [A_vel, b_vel] = compute_velocity_constraints(V_base, Bv, params)
    % 输入: 
    %   V_base: 惯性滑行预测速度 (2*N*H x 1)
    %   Bv: 速度响应矩阵 (2*N*H x 2*N*H)
    %   params: 包含 n, h_ac, v_max

    N = params.n;
    H = params.h_ac;
    v_max = params.v_max;
    
    % 线性化逼近参数：用 M 条线段逼近一个圆
    M = 8; % 边数越多越接近圆，计算量也越大
    theta = linspace(0, 2*pi, M+1);
    theta(end) = []; % 去掉重复的 2pi

    % 初始化 A 和 b
    % 每个智能体、每一步、每个方向都需要一条约束
    A_vel = zeros(N * H * M, 2 * N * H);
    b_vel = zeros(N * H * M, 1);

    row = 1;
    for k = 1:H % 遍历预测步
        for i = 1:N % 遍历智能体
            % 获取当前智能体在第 k 步的速度在 V_base 中的索引
            idx = (k-1)*2*N + (i-1)*2 + (1:2);
            
            % 提取该智能体这一拍对应的 Bv 子矩阵
            % V_future_ik = V_base(idx) + Bv(idx, :) * U
            
            for m = 1:M % 遍历多边形的每一条边
                % 方向向量：[cos(theta), sin(theta)]
                n_dir = [cos(theta(m)), sin(theta(m))];
                
                % 线性不等式: n_dir * V_future <= v_max
                % 展开: n_dir * (V_base(idx) + Bv(idx, :) * U) <= v_max
                % 变形为 A*U <= b:
                % (n_dir * Bv(idx, :)) * U <= v_max - n_dir * V_base(idx)
                
                A_vel(row, :) = n_dir * Bv(idx, :);
                b_vel(row, 1) = v_max - n_dir * V_base(idx);
                row = row + 1;
            end
        end
    end
end