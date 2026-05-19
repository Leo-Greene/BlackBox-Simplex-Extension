%% verify_pipeline_once.m
% A single-run diagnostic script to run the PrSBC-NN control pipeline once.
% It prints detailed state variables, network outputs, Jacobian metrics,
% and QP solver execution diagnostics to help diagnose safety/collision issues.

clc;
clear;
clear functions;
close all;

fprintf('====================================================================\n');
fprintf('                START DIAGNOSTIC PIPELINE RUN                       \n');
fprintf('====================================================================\n\n');

%% 1. Path Configuration & Dependencies
PROJECT_ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(PROJECT_ROOT, 'controllers', 'ac', 'controller_cmpc_2d_quadprog')));
addpath(genpath(fullfile(PROJECT_ROOT, 'controllers', 'bc', 'safety_controller_quadprog')));
addpath(genpath(fullfile(PROJECT_ROOT, 'controllers', 'PrSBC_filter')));
addpath(genpath(fullfile(PROJECT_ROOT, 'decision_module')));
addpath(genpath(fullfile(PROJECT_ROOT, 'extended_BBS')));
addpath(fullfile(PROJECT_ROOT, 'common'));
addpath(genpath(fullfile(PROJECT_ROOT, 'experiment', 'dynamics')));
addpath(genpath(fullfile(PROJECT_ROOT, 'experiment', 'utilities')));
addpath(genpath(fullfile(PROJECT_ROOT, 'experiment', '1_data_collection')));
addpath(fullfile(PROJECT_ROOT, 'experiment', 'other'));

%% 2. Find and Load LATEST Trained NN Model
TRAIN_OUT = fullfile(PROJECT_ROOT, 'experiment', '2_train', 'out');
d = dir(fullfile(TRAIN_OUT, '20*'));
if isempty(d)
    error('No model directory found in %s. Please train a model first.', TRAIN_OUT);
end
[~, sorted_idx] = sort({d.name});
LATEST_MODEL_DIR = fullfile(TRAIN_OUT, d(sorted_idx(end)).name);

ONNX_PATH = fullfile(LATEST_MODEL_DIR, 'residual_model.onnx');
STATS_PATH = fullfile(LATEST_MODEL_DIR, 'scaling_stats.json');
METRICS_PATH = fullfile(LATEST_MODEL_DIR, 'validation_metrics.json');

fprintf('--> Loading Neural Network Model From:\n');
fprintf('    Model Dir: %s\n', LATEST_MODEL_DIR);
fprintf('    ONNX File: %s\n', ONNX_PATH);

% Compile ONNX Function
RUN_OUT_ROOT = fullfile(PROJECT_ROOT, 'traj', 'step4_integrate', 'diagnose_run');
if ~exist(RUN_OUT_ROOT, 'dir')
    mkdir(RUN_OUT_ROOT);
end
onnx_func_path = fullfile(RUN_OUT_ROOT, 'residual_net');
params_onnx = importONNXFunction(ONNX_PATH, onnx_func_path);
addpath(RUN_OUT_ROOT);

% Load Scaling Stats
fid = fopen(STATS_PATH);
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
stats_raw = jsondecode(str);

stats.X_mean = reshape(double(stats_raw.X_mean), 1, []); 
stats.X_std  = reshape(double(stats_raw.X_std), 1, []);
stats.U_mean = reshape(double(stats_raw.U_mean), 1, []);
stats.U_std  = reshape(double(stats_raw.U_std), 1, []);
stats.R_mean = reshape(double(stats_raw.R_mean), 1, []);
stats.R_std  = reshape(double(stats_raw.R_std), 1, []);

learned_model.func = @residual_net;
learned_model.params_onnx = params_onnx;
learned_model.stats = stats;

% Load Data-Driven Process Noise Variance
if exist(METRICS_PATH, 'file')
    fid = fopen(METRICS_PATH);
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    metrics = jsondecode(str);
    residual_variance = metrics.residual_variance;
else
    residual_variance = 0.05^2;
end
fprintf('--> Loaded Data-Driven Process Noise Variance: %.6f\n\n', residual_variance);

%% 3. Setup Case Config and Overrides
cfg = struct();
cfg.case_id = 5; % Choose a specific diagnostic case
cfg.seed = 42;
cfg.enable_plot = false; % Disable plots to keep it clean
cfg.save_mat = false;
cfg.output_root = RUN_OUT_ROOT;

cfg.params_overrides = struct();
cfg.params_overrides.steps = 60; % Run full 60 steps!
cfg.params_overrides.use_learned_dynamics = true;
cfg.params_overrides.use_prsbc_filter = true;
cfg.params_overrides.prsbc_use_nn = true;
cfg.params_overrides.learned_model = learned_model;
cfg.params_overrides.residual_variance = residual_variance;
cfg.params_overrides.control_noise_std = 0; % 禁用测试时的控制噪声！
cfg.params_overrides.explore_noise_std = 0; % 禁用测试时的探索噪声！
cfg.params_overrides.verbose = false; % Disable noisy log prints to keep table clean

%% 4. Execute Simulation
fprintf('--> Launching Full 60-Step Simulation...\n');
try
    [traj, run_info] = run_bb_reverse_once(cfg);
    fprintf('\n--> Simulation completed successfully! Performing state prediction analysis...\n\n');
    
    % 5. Post-run NN vs Nominal Prediction Analysis
    dt = traj.params.dt;
    alpha_x = traj.params.alpha_x;
    alpha_v = traj.params.alpha_v;
    n = traj.params.n;
    steps = traj.params.steps;
    
    err_nom_all = zeros(steps, 1);
    err_nn_all = zeros(steps, 1);
    
    fprintf('=========================================================================\n');
    fprintf('     STEP-BY-STEP PREDICTION ACCURACY REPORT (Average across all %d agents)\n', n);
    fprintf('=========================================================================\n');
    fprintf('  Step  |  Nominal Mismatch Error (m)  |  NN-Corrected Mismatch Error (m)  |  Error Reduction  \n');
    fprintf('-------------------------------------------------------------------------\n');
    
    for t = 1:steps
        % Current physical state
        pos_t = [traj.x(t, :); traj.y(t, :)];
        vel_t = [traj.vx(t, :); traj.vy(t, :)];
        u_t = [traj.ax_applied(t, :); traj.ay_applied(t, :)];
        
        % Mismodeled Nominal Prediction (Zero-order + nominal input)
        p_nom = pos_t + alpha_x * (alpha_v * vel_t) * dt + alpha_x * alpha_v * dt^2 * u_t;
        
        % NN-Corrected Prediction
        v_obs = vel_t'; % n x 2
        u_prev = u_t';  % n x 2
        [res_all, ~] = predict_residual_and_jacobian(v_obs, u_prev, learned_model.func, learned_model.params_onnx, learned_model.stats);
        r_pos = res_all(:, 1:2)'; % 2 x n position residual
        p_pred = p_nom + r_pos;
        
        % Actual next state (Ground Truth)
        pos_next = [traj.x(t+1, :); traj.y(t+1, :)];
        
        % L2 error (Euclidean distance) per agent
        err_nom = sqrt(sum((p_nom - pos_next).^2, 1));
        err_nn = sqrt(sum((p_pred - pos_next).^2, 1));
        
        avg_err_nom = mean(err_nom);
        avg_err_nn = mean(err_nn);
        
        err_nom_all(t) = avg_err_nom;
        err_nn_all(t) = avg_err_nn;
        
        reduction = (avg_err_nom - avg_err_nn) / avg_err_nom * 100;
        
        fprintf('   %02d   |           %.6f m          |             %.6f m            |      %5.1f%%      \n', ...
            t, avg_err_nom, avg_err_nn, reduction);
    end
    fprintf('-------------------------------------------------------------------------\n');
    fprintf('  OVERALL AVERAGE POSITION ERRORS:\n');
    fprintf('  - Mismodeled Nominal Model Error : %.6f meters\n', mean(err_nom_all));
    fprintf('  - NN-Corrected Model Error       : %.6f meters\n', mean(err_nn_all));
    fprintf('  - Average Error Reduction        : %.2f%%\n', (mean(err_nom_all) - mean(err_nn_all)) / mean(err_nom_all) * 100);
    fprintf('  - Random Physical Process Noise  : %.6f meters (Theoretical Limit)\n', traj.params.epsilon_w_pos);
    fprintf('=========================================================================\n\n');
    
    % Print Collisions Info
    collisions = check_collision(traj);
    if isempty(collisions)
        fprintf('========== COLLISION REPORT ==========\n');
        fprintf('  [SAFE] Absolutely NO collisions occurred during the 60 steps!\n');
        fprintf('======================================\n\n');
    else
        fprintf('========== COLLISION REPORT ==========\n');
        fprintf('  [WARNING] %d collision events detected during the 60 steps!\n', length(collisions));
        fprintf('======================================\n\n');
    end

catch ME
    fprintf('\n--> [CRITICAL ERROR] Simulation or Analysis Failed!\n');
    fprintf('    Error Message: %s\n', ME.message);
    fprintf('    Error Stack:\n');
    for k = 1:length(ME.stack)
        disp(ME.stack(k));
    end
end
