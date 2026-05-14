function [ pos, vel ] = plant_dynamics( pos, vel, acc, params )
%% 系统在实际演化时添加物理过程噪声

% 1. 基于加速度更新名义速度
vel = vel + params.dt*acc;

%% Extended BBS (速度边界裁剪)
% actives = params.actvie_agents;
for j = 1:params.n
%   if actives(j) == 1
        if params.predator && j == params.n
           if norm(vel(:,j)) > params.pFactor * params.vmax
                vel(:,j) = (params.pFactor * params.vmax / norm( vel(:,j) )) * vel(:,j);
           end
           continue;
        end
        if norm(vel(:,j)) > params.vmax
            vel(:,j) = (params.vmax/norm( vel(:,j) )) * vel(:,j);
        end
%   end
end

%% Extended BBS (基于裁剪后的速度更新名义位置)
for i=1:params.n
%   if actives(i) == 1
        pos(:, i) = pos(:, i) + params.dt*vel(:, i);
%   end
end

%% ================== 新增：注入物理过程噪声 w(k) ==================

for i = 1:params.n
    % --- 生成位置噪声向量 w_i (确保在半径为 epsilon_i 的圆内) ---
    theta = 2 * pi * rand(); % 随机方向 (0 到 2*pi)
    % 注意：半径必须用 sqrt(rand()) 才能保证圆内分布均匀，直接 rand() 会导致圆心密集
    r = params.epsilon_w_pos * sqrt(rand());
    w_i_pos = [r * cos(theta); r * sin(theta)];
    % Add noise to each robot's position
    pos(:, i) = pos(:, i) + w_i_pos;
end

%% =================================================================

end