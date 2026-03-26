function [residual, x_norm, u_norm] = predict_residual(x, u, model_func, params_onnx, scaleStats)
% PREDICT_RESIDUAL Standard wrapper for the current model architecture.
%
% This is the primary inference function for the robust multi-step model.

    x = double(reshape(x, 1, 60));
    u = double(reshape(u, 1, 30));
    
    x_mean = double(reshape(scaleStats.X_mean, 1, 60));
    x_std  = double(reshape(scaleStats.X_std, 1, 60));
    u_mean = double(reshape(scaleStats.U_mean, 1, 30));
    u_std  = double(reshape(scaleStats.U_std, 1, 30));
    
    x_norm = (x - x_mean) ./ (x_std + 1e-8);
    u_norm = (u - u_mean) ./ (u_std + 1e-8);
    
    try
        res_norm_raw = model_func(single(x_norm), single(u_norm), params_onnx);
        res_norm = double(res_norm_raw(1, :));
    catch ME
        error('Error during ONNX model inference: %s', ME.message);
    end
    
    r_mean = double(reshape(scaleStats.R_mean, 1, 60));
    r_std  = double(reshape(scaleStats.R_std, 1, 60));
    residual = double(res_norm) .* r_std + r_mean;
end