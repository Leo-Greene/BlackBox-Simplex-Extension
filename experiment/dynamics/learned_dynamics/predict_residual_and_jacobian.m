function [residual, J_u] = predict_residual_and_jacobian(v_obs, u_prev, model_func, params_onnx, scaleStats)
% PREDICT_RESIDUAL_AND_JACOBIAN Computes the residual and its Jacobian w.r.t u
%
% Inputs:
%   v_obs: [n x 2] velocity observation (vx, vy)
%   u_prev: [n x 2] previous control input (ax, ay)
%   model_func: compiled ONNX function handle (@residual_net)
%   params_onnx: ONNX parameters
%   scaleStats: normalization stats
%
% Outputs:
%   residual: [n x 4] residual vector [dx, dy, dvx, dvy]
%   J_u: [4 x 2 x n] Jacobian tensor of residual w.r.t u_prev

    n = size(v_obs, 1);
    
    % 1. Base Residual Prediction
    residual = predict_residual_core(v_obs, u_prev, model_func, params_onnx, scaleStats, n);
    
    % 2. Finite Difference for Jacobian
    delta = 1e-4;
    J_u = zeros(4, 2, n);
    
    for j = 1:2
        u_pert = u_prev;
        u_pert(:, j) = u_pert(:, j) + delta;
        res_pert = predict_residual_core(v_obs, u_pert, model_func, params_onnx, scaleStats, n);
        
        for i = 1:n
            J_u(:, j, i) = (res_pert(i, :) - residual(i, :))' / delta;
        end
    end
end

function res = predict_residual_core(v, u, model_func, params_onnx, stats, n)
    % 1. Reshape stats to ensure 1D vectors
    x_mean = double(reshape(stats.X_mean, 1, []));
    x_std  = double(reshape(stats.X_std, 1, []));
    u_mean = double(reshape(stats.U_mean, 1, []));
    u_std  = double(reshape(stats.U_std, 1, []));
    r_mean = double(reshape(stats.R_mean, 1, []));
    r_std  = double(reshape(stats.R_std, 1, []));

    % 2. Extract agent-specific slices
    vx_mean = x_mean(2*n+1 : 3*n);
    vx_std  = x_std(2*n+1 : 3*n);
    vy_mean = x_mean(3*n+1 : 4*n);
    vy_std  = x_std(3*n+1 : 4*n);

    ax_mean = u_mean(1 : n);
    ax_std  = u_std(1 : n);
    ay_mean = u_mean(n+1 : 2*n);
    ay_std  = u_std(n+1 : 2*n);

    % 3. Normalize inputs
    vx_n = (v(:, 1)' - vx_mean) ./ (vx_std + 1e-8);
    vy_n = (v(:, 2)' - vy_mean) ./ (vy_std + 1e-8);
    ax_n = (u(:, 1)' - ax_mean) ./ (ax_std + 1e-8);
    ay_n = (u(:, 2)' - ay_mean) ./ (ay_std + 1e-8);

    v_norm = [vx_n(:), vy_n(:)];
    u_norm = [ax_n(:), ay_n(:)];

    % 4. Model Inference
    res_norm = zeros(n, 4);
    for i = 1:n
        out = model_func(single(v_norm(i,:)), single(u_norm(i,:)), params_onnx);
        res_norm(i,:) = double(out);
    end
    
    % 5. Denormalize outputs
    dx_mean = r_mean(1 : n);           dx_std = r_std(1 : n);
    dy_mean = r_mean(n+1 : 2*n);       dy_std = r_std(n+1 : 2*n);
    dvx_mean = r_mean(2*n+1 : 3*n);    dvx_std = r_std(2*n+1 : 3*n);
    dvy_mean = r_mean(3*n+1 : 4*n);    dvy_std = r_std(3*n+1 : 4*n);

    dx = res_norm(:, 1) .* dx_std(:) + dx_mean(:);
    dy = res_norm(:, 2) .* dy_std(:) + dy_mean(:);
    dvx = res_norm(:, 3) .* dvx_std(:) + dvx_mean(:);
    dvy = res_norm(:, 4) .* dvy_std(:) + dvy_mean(:);
    
    res = [dx, dy, dvx, dvy];
end
