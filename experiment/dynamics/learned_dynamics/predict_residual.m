function [residual, x_norm, u_norm] = predict_residual(x, u, model_func, params_onnx, scaleStats)
% PREDICT_RESIDUAL Standard wrapper for the current model architecture.
%
% This is the primary inference function for the robust multi-step model.

    x = double(reshape(x, 1, []));
    u = double(reshape(u, 1, []));
    n = numel(x) / 4;

    x_mean = double(reshape(scaleStats.X_mean, 1, []));
    x_std  = double(reshape(scaleStats.X_std, 1, []));
    u_mean = double(reshape(scaleStats.U_mean, 1, []));
    u_std  = double(reshape(scaleStats.U_std, 1, []));

    x_norm = (x - x_mean) ./ (x_std + 1e-8);
    u_norm = (u - u_mean) ./ (u_std + 1e-8);

    vx_n = x_norm(2*n+1 : 3*n);
    vy_n = x_norm(3*n+1 : 4*n);
    ax_n = u_norm(1 : n);
    ay_n = u_norm(n+1 : 2*n);

    v_flat = [vx_n(:), vy_n(:)];
    u_flat = [ax_n(:), ay_n(:)];

    try
        res_norm = zeros(n, 4);
        for i = 1:n
            res_norm_raw = model_func(single(v_flat(i,:)), single(u_flat(i,:)), params_onnx);
            res_norm(i,:) = double(res_norm_raw);
        end
    catch ME
        error('Error during ONNX model inference: %s', ME.message);
    end

    r_mean = double(reshape(scaleStats.R_mean, 1, []));
    r_std  = double(reshape(scaleStats.R_std, 1, []));
    dx_mean = r_mean(1 : n);           dx_std = r_std(1 : n);
    dy_mean = r_mean(n+1 : 2*n);       dy_std = r_std(n+1 : 2*n);
    dvx_mean = r_mean(2*n+1 : 3*n);    dvx_std = r_std(2*n+1 : 3*n);
    dvy_mean = r_mean(3*n+1 : 4*n);    dvy_std = r_std(3*n+1 : 4*n);

    dx = res_norm(:, 1) .* dx_std(:) + dx_mean(:);
    dy = res_norm(:, 2) .* dy_std(:) + dy_mean(:);
    dvx = res_norm(:, 3) .* dvx_std(:) + dvx_mean(:);
    dvy = res_norm(:, 4) .* dvy_std(:) + dvy_mean(:);

    residual = [dx.', dy.', dvx.', dvy.'];
end