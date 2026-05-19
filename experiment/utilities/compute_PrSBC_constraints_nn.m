%% compute_PrSBC_constraints
% Computes the constraints for a probabilistic Control Barrier Function (PrSBC) for a group of agents.

function [A_sbc, b_sbc] = compute_PrSBC_constraints_nn(pos, vel, params)
    n = params.n;
    h_ac = params.h_ac;
    dim = 2;
    N_vars = dim * n * h_ac; 
    dt = params.dt;

    % 1. Parse Dynamics Parameters
    alpha_x = params.alpha_x;
    alpha_v = params.alpha_v;
    G_nom = alpha_x * alpha_v * dt^2 * eye(2);

    % 2. Precompute F_eff and G_eff for all agents (O(N))
    F_eff = zeros(2, n);
    G_eff = zeros(2, 2, n);
    
    if isfield(params, 'use_learned_dynamics') && params.use_learned_dynamics
        % Prepare inputs for Neural Network
        v_obs = vel'; % [n x 2]
        if isfield(params, 'u_cmd')
            u_prev = params.u_cmd'; % Use previous/commanded control [n x 2]
        else
            u_prev = zeros(n, 2);
        end
        
        model_func = params.learned_model.func;
        params_onnx = params.learned_model.params_onnx;
        stats = params.learned_model.stats;
        
        % Step 1: Call highly optimized Predictor & Jacobian
        [res_all, J_u_all] = predict_residual_and_jacobian(v_obs, u_prev, model_func, params_onnx, stats);
        
        % Data-Driven Process Noise from Validation Metrics
        if isfield(params, 'residual_variance') && isfield(params.learned_model, 'stats')
            % FIX B (2026-05-19): Use RMS of ALL 2n position-residual std-devs as the
            % de-normalization scale, instead of the single R_std(1) scalar.
            %
            % Rationale:
            %   - R_label layout: [dx_1..n, dy_1..n, dvx_1..n, dvy_1..n]
            %   - PrSBC epsilon_w is a 2-D position uncertainty radius (meters).
            %   - The position components occupy indices 1:2n in R_std.
            %   - Taking the RMS over all 2n elements gives the isotropic L2-average
            %     noise amplitude, consistent with the scalar Euclidean expansion in
            %     the CBF formula: (R_safe + epsilon_total)^2.
            %   - Using only R_std(1) was agent-1-x-biased and not representative
            %     of the full population of pairwise distance uncertainties.
            R_std_vec = params.learned_model.stats.R_std(:);
            R_std_pos = R_std_vec(1 : 2*n);          % [dx_1..n, dy_1..n]
            epsilon_w = sqrt(params.residual_variance) * sqrt(mean(R_std_pos.^2));
        else
            persistent has_warned_variance_fallback;
            if isempty(has_warned_variance_fallback)
                warning('[FALLBACK] [PrSBC NN] Missing residual_variance or stats in params! Falling back to default epsilon_w = 0.05.');
                has_warned_variance_fallback = true;
            end
            epsilon_w = 0.05; % fallback
        end
        
        for i = 1:n
            p_t = pos(:, i);
            v_t = vel(:, i);
            u_i = u_prev(i, :)';
            
            % Nominal Base (Zero Control)
            F_base = p_t + alpha_x * (alpha_v * v_t) * dt;
            
            % Residual terms for position (dx, dy are indices 1, 2)
            r_pos = res_all(i, 1:2)';
            J_pos_u = J_u_all(1:2, :, i); % 2x2 Jacobian for position
            
            % Effective Base Drift and Control Gain
            F_eff(:, i) = F_base + r_pos - J_pos_u * u_i;
            G_eff(:, :, i) = G_nom + J_pos_u;
        end
    else
        % Fallback to pure nominal mismatch
        fprintf('[FALLBACK] [PrSBC NN] Fallback to pure nominal mismatch.\n');
        epsilon_w = params.epsilon_w_pos; 
        for i = 1:n
            p_t = pos(:, i);
            v_t = vel(:, i);
            F_eff(:, i) = p_t + alpha_x * (alpha_v * v_t) * dt;
            G_eff(:, :, i) = G_nom;
        end
    end

    % 3. Initialize QP Constraints
    num_constraints = nchoosek(n, 2);
    A_sbc = zeros(num_constraints, N_vars);
    b_sbc = zeros(num_constraints, 1);

    gamma = params.gamma;
    Confidence = params.confidence;
    Sensing_Range = params.sensing_range;

    % Noise parameters
    sigma_obs = params.sigma_obs_pos; 
    sigma_vel = params.sigma_obs_vel; 
    sigma_total_sq = 2 * (sigma_obs^2 + (sigma_vel * dt)^2 + epsilon_w^2);
    sigma_total = sqrt(sigma_total_sq);
    quantile_1D = sqrt(2) * erfinv(2 * Confidence - 1);
    epsilon_total = sigma_total * quantile_1D;

    persistent has_printed_probe;
    if isempty(has_printed_probe)
        fprintf('[PrSBC NN] Active! Using Data-Driven epsilon_w: %.6f (Fallback was 0.05). Total Expansion: %.6f\n', epsilon_w, epsilon_total);
        has_printed_probe = true;
    end

    % 4. Pairwise Collision Checking Loop (O(N^2))
    count = 1;
    for i = 1:n-1
        for j = i+1:n
            dist_current = norm(pos(:, i) - pos(:, j));
            if dist_current > Sensing_Range
                A_sbc(count, :) = 0;
                b_sbc(count) = 1e6;
                count = count + 1;
                continue;
            end

            h_t = dist_current^2 - params.R_safe^2;

            % Step 3: Compute exact first-order separation vectors
            p_next_base_new = F_eff(:, i) - F_eff(:, j);
            norm_p_next = norm(p_next_base_new);
            
            if norm_p_next < 1e-4
                norm_p_next = 1e-4;
                p_next_base_new = [1e-4; 0];
            end

            % QP Matrix Construction
            idx_i = (2*i - 1) : (2*i);
            idx_j = (2*j - 1) : (2*j);

            A_sbc(count, idx_i) = -2 * p_next_base_new' * G_eff(:, :, i);
            A_sbc(count, idx_j) = -2 * p_next_base_new' * (-G_eff(:, :, j));

            b_val = norm_p_next^2 - (params.R_safe + epsilon_total)^2 - (1 - gamma) * h_t;
            b_sbc(count) = b_val;
            
            if isfield(params, 'verbose') && params.verbose && (dist_current < params.R_safe * 1.3)
                fprintf('\n================== [PrSBC Verbose Pairwise Probe] ==================\n');
                fprintf('  - Pairwise: Agent %d <-> Agent %d\n', i, j);
                fprintf('  - Current Dist: %.4f | R_safe: %.4f | h(t): %.4f\n', dist_current, params.R_safe, h_t);
                fprintf('  - F_eff_i: [%.4f; %.4f] | F_eff_j: [%.4f; %.4f]\n', F_eff(1,i), F_eff(2,i), F_eff(1,j), F_eff(2,j));
                fprintf('  - G_eff_i: [%.4f, %.4f; %.4f, %.4f]\n', G_eff(1,1,i), G_eff(1,2,i), G_eff(2,1,i), G_eff(2,2,i));
                fprintf('  - G_eff_j: [%.4f, %.4f; %.4f, %.4f]\n', G_eff(1,1,j), G_eff(1,2,j), G_eff(2,1,j), G_eff(2,2,j));
                fprintf('  - u_prev_i: [%.4f; %.4f] | u_prev_j: [%.4f; %.4f]\n', u_prev(i,1), u_prev(i,2), u_prev(j,1), u_prev(j,2));
                fprintf('  - p_next_base_new (F_eff diff): [%.4f; %.4f] | norm: %.4f\n', p_next_base_new(1), p_next_base_new(2), norm_p_next);
                fprintf('  - A_coeff_i: [%.4f, %.4f] | A_coeff_j: [%.4f, %.4f]\n', A_sbc(count, idx_i), A_sbc(count, idx_j));
                fprintf('  - b_val: %.4f | epsilon_total (Noise Expansion): %.4f\n', b_val, epsilon_total);
                fprintf('====================================================================\n\n');
            end
            
            count = count + 1;
        end
    end
end