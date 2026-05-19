function [x_next, residual] = dynamics_learned(x_nominal, x_residual, u, model_func, params_onnx, scaleStats, params)
% DYNAMICS_LEARNED Learned dynamics wrapper for Decision Module rollout
% formula: x_next = f_nominal(x, u) + r_phi(x, u)
%
% Inputs:
%   x_nominal   - Nominal model state input [1 x 4n]
%   x_residual  - Residual model state input [1 x 4n]
%   u           - Current action [1 x 2n]
%   model_func  - Function handle for ONNX code (e.g. @residual_net)
%   params_onnx - Parameter struct from importONNXFunction
%   scaleStats  - Scaling statistics
%   params      - Physical parameters struct (must contain dt, vmax, n)
%
% Outputs:
%   x_next     - Predicted next state [1 x 4n]
%   residual   - Network predicted residual [1 x 4n]

    % 1. 计算标称模型预测 (Nominal Prediction)
    % 严格对齐 dynamics.m 的积分逻辑：先更新速度(含限速)，再更新位置
    dt = params.dt; 
    vmax = params.vmax; 
    n = params.n;
    alpha_v = 1.0;
    alpha_x = 1.0;
    if isfield(params, 'alpha_v'), alpha_v = params.alpha_v; end
    if isfield(params, 'alpha_x'), alpha_x = params.alpha_x; end
    predator = false;
    pFactor = 1.0;
    if isfield(params, 'predator'), predator = params.predator; end
    if isfield(params, 'pFactor'), pFactor = params.pFactor; end
    
    % 确保输入维度正确 (自动适配 n)
    x_nominal = double(reshape(x_nominal, 1, 4*n));
    x_residual = double(reshape(x_residual, 1, 4*n));
    u = double(reshape(u, 1, 2*n));
    
    % 提取分量 (动态索引)
    px = x_nominal(1 : n);           py = x_nominal(n+1 : 2*n);
    vx = x_nominal(2*n+1 : 3*n);     vy = x_nominal(3*n+1 : 4*n);
    ax = u(1 : n);           ay = u(n+1 : 2*n);
    
    % --- Step A: 速度更新 (带物理限速) ---
    vx_next = alpha_v * (vx + ax * dt);
    vy_next = alpha_v * (vy + ay * dt);
    
    for j = 1:n
        v_mag = sqrt(vx_next(j)^2 + vy_next(j)^2);
        vmax_j = vmax;
        if predator && j == n
            vmax_j = vmax * pFactor;
        end
        if v_mag > vmax_j
            scale = vmax_j / v_mag;
            vx_next(j) = vx_next(j) * scale;
            vy_next(j) = vy_next(j) * scale;
        end
    end
    
    % --- Step B: 位置更新 (使用更新后的速度) ---
    px_next = px + alpha_x * (vx_next * dt);
    py_next = py + alpha_x * (vy_next * dt);
    
    x_nom = [px_next, py_next, vx_next, vy_next];
    
    % 2. 计算残差预测 (使用封装好的归一化推理器)
    residual = predict_residual(x_residual, u, model_func, params_onnx, scaleStats);
    
    % 3. 结果合成
    x_next = x_nom + residual;
    
    % 额外的保护：叠加残差后再次强制限速，确保网络输出不会违反物理常识
    vx_final = x_next(2*n+1 : 3*n);
    vy_final = x_next(3*n+1 : 4*n);
    for j = 1:n
        v_mag = sqrt(vx_final(j)^2 + vy_final(j)^2);
        vmax_j = vmax;
        if predator && j == n
            vmax_j = vmax * pFactor;
        end
        if v_mag > vmax_j
            scale = vmax_j / v_mag;
            vx_final(j) = vx_final(j) * scale;
            vy_final(j) = vy_final(j) * scale;
        end
    end
    x_next(2*n+1 : 4*n) = [vx_final, vy_final];

end
