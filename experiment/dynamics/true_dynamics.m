function [pos_next, vel_next] = true_dynamics(pos, vel, acc, params)
%% true_dynamics() - 当前阶段的真实动力学（deterministic residual baseline）
%
% 目标：
%   构造一个与名义模型 f(x,u) 存在“系统性偏差”的真实 Plant，
%   但暂不显式加入随机噪声，也不引入额外内部记忆状态。
%
% 思路：
%   x_{k+1}^{true} = f(x_k,u_k) + d(x_k,u_k)
%
% 其中 d(x_k,u_k) 由以下因素构成：
%   1) 参数失配（如有效加速度缩放）
%   2) 速度阻尼（未建模摩擦）
%   3) 小幅状态相关非线性漂移

dt = params.dt;

%% 1) 参数失配：真实执行加速度 != 名义输入加速度
acc_actual = acc;

if isfield(params, 'acc_scale')
    acc_actual = params.acc_scale .* acc_actual;   % 例如 0.9 * acc
end

if isfield(params, 'acc_bias')
    acc_actual = acc_actual + params.acc_bias;     % 常值偏差
end

%% 2) 未建模阻尼 / 摩擦
vel_next = vel + dt * acc_actual;

if isfield(params, 'damping') && params.damping > 0
    vel_next = vel_next - dt * params.damping * vel;
end

%% 3) 与名义模型一致的速度限制
for j = 1:params.n
    if params.predator && j == params.n
        vmax_j = params.pFactor * params.vmax;
    else
        vmax_j = params.vmax;
    end

    nv = norm(vel_next(:, j));
    if nv > vmax_j
        vel_next(:, j) = (vmax_j / nv) * vel_next(:, j);
    end
end

%% 4) 位置更新：名义项 + 小幅非线性漂移
pos_next = pos + dt * vel_next;

if isfield(params, 'nonlinear_drift') && params.nonlinear_drift
    for i = 1:params.n
        drift = 0.001 * [sin(0.1 * pos(1,i)); cos(0.1 * pos(2,i))];
        pos_next(:, i) = pos_next(:, i) + drift;
    end
end

end