function [P_base, V_base, Bp, Bv] = compute_nominal_predictions(pos, vel, params)

% 假设当前状态：
% pos: 2 x n 矩阵
% vel: 2 x n 矩阵

N = params.n;
H = params.h_ac;
dt = params.dt;

% 将当前 pos 和 vel 拉直成列向量 (2n x 1)
%按照 MATLAB 的展开逻辑，展开后的分布如下：
% 索引 1 & 2：第 1 个智能体的坐标 $[x_1; y_1]$。  
% 索引 3 & 4：第 2 个智能体的坐标 $[x_2; y_2]$。  
% 索引 $2i-1$ & $2i$：第 $i$ 个智能体的坐标 $[x_i; y_i]$。
p0 = pos(:); 
v0 = vel(:);

% --- 1. 计算基础轨迹 (仅靠初速度惯性滑行) ---
% 初始化大向量
V_base = zeros(2 * N * H, 1);
P_base = zeros(2 * N * H, 1);

for k = 1:H
    % 基础速度：每一拍都是 v0
    V_base((k-1)*2*N + 1 : k*2*N) = v0;
    
    % 基础位置：根据“先改速度”逻辑，第 k 拍的位置
    % 注意：即便 a=0，第 k 步的位置也是在 p0 基础上累加了 k 个 v0*dt
    P_base((k-1)*2*N + 1 : k*2*N) = p0 + k * dt * v0;
end

% --- 2. 计算控制响应矩阵 Bp 和 Bv (下三角结构) ---
Bp = zeros(2 * N * H, 2 * N * H);
Bv = zeros(2 * N * H, 2 * N * H);

I_2N = eye(2 * N); % 预先生成 2N x 2N 的单位阵，提高效率

% 双重循环构建下三角分块矩阵
for i = 1:H          % i 表示当前预测的时刻 (行块)
    for j = 1:i      % j 表示过去施加控制指令的时刻 (列块)
        
        % 行索引和列索引
        row_idx = (i-1)*2*N + 1 : i*2*N;
        col_idx = (j-1)*2*N + 1 : j*2*N;
        
        % 速度响应：只要控制指令施加了，对未来速度的影响系数始终是 dt
        Bv(row_idx, col_idx) = I_2N * dt;
        
        % 位置响应：控制指令施加得越早，累积的位置偏移越大
        % 作用时长步数为 (i - j + 1)
        Bp(row_idx, col_idx) = I_2N * (i - j + 1) * dt^2;
        
    end
end