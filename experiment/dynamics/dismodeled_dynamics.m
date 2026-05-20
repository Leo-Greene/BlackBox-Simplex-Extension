function [pos_next, vel_next] = dismodeled_dynamics(pos, vel, acc, params)
%% dismodeled_dynamics() - 专门供 DM 和 PrSBC 使用的、带有 Alpha 参数失配的名义动力学模型
%
% 核心逻辑：
%   1. 剔除一切物理世界的偏置(acc_bias)和阻尼(damping)，保持数学公式的纯净。
%   2. 严格遵循无绝对坐标污染的“平移不变性”半隐式欧拉积分。
%   3. 与真实物理引擎保持一致的物理最高限速截断。

dt = params.dt;
n = params.n;

% 提取错配系数，若 params 里没给，则默认不发生错配（即为 1.0）
alpha_v = 1.0;
alpha_x = 1.0;
if isfield(params, 'alpha_v'), alpha_v = params.alpha_v; end
if isfield(params, 'alpha_x'), alpha_x = params.alpha_x; end

%% 1) 速度级错配更新
% v_{t+1}^{nom} = \alpha_v * (v_t + a_t * dt)
vel_next = alpha_v .* (vel + acc * dt);

%% 2) 物理最高限速截断 (与物理世界对齐，保留 predator 动态限制)
vmax = 2.0;
pFactor = 1.40;
predator = 0;

if isfield(params, 'vmax'), vmax = params.vmax; end
if isfield(params, 'pFactor'), pFactor = params.pFactor; end
if isfield(params, 'predator'), predator = params.predator; end

for j = 1:n
    if predator && j == n
        vmax_j = pFactor * vmax;
    else
        vmax_j = vmax;
    end

    nv = norm(vel_next(:, j));
    if nv > vmax_j
        vel_next(:, j) = (vmax_j / nv) * vel_next(:, j);
    end
end

%% 3) 位置级错配更新 (严格防止绝对坐标 pos 受到 alpha_x 污染)
% x_{t+1}^{nom} = x_t + \alpha_x * (v_{t+1}^{nom} * dt)
pos_next = pos + alpha_x .* (vel_next * dt);

end