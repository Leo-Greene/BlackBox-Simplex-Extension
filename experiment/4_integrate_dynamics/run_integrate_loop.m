% run_integrate_loop.m
% Integrated Evaluation of Learned Dynamics Model (Decision Module Multi-step)
clc;
clear;
close all;

%% 1. 路径设置 Path Setup
PROJECT_ROOT = fullfile(fileparts(mfilename('fullpath')), '..', '..');
addpath(genpath(fullfile(PROJECT_ROOT, 'decision_module')));
addpath(fullfile(PROJECT_ROOT, 'common'));
addpath(fullfile(PROJECT_ROOT, 'experiment', 'other')); % for check_collision
addpath(genpath(fullfile(PROJECT_ROOT, 'experiment', 'dynamics', 'learned_dynamics')));
addpath(fullfile(PROJECT_ROOT, 'experiment', '1_data_collection')); % run_bb_reverse_once
addpath(PROJECT_ROOT); % for gen_init_bb.m and other root tools

report_timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
RUN_OUT_ROOT = fullfile(PROJECT_ROOT, 'traj', 'step4_integrate', report_timestamp);
if ~exist(RUN_OUT_ROOT, 'dir'), mkdir(RUN_OUT_ROOT); end

%% 3. 并行设置 Parallel Setup
force_parallel = true;
requested_workers = 0; 
use_parallel = force_parallel && license('test', 'Distrib_Computing_Toolbox');

if use_parallel
    pool = gcp('nocreate');
    if isempty(pool)
        if requested_workers > 0
            parpool(requested_workers);
        else
            parpool;
        end
    end
end

%% 4. 加载神经网络模型和统计数据 Load Neural Network Model & Stats
fprintf('--> Initializing Learned Dynamics Module...\n');
TRAIN_OUT = fullfile(PROJECT_ROOT, 'experiment', '2_train', 'out');
d = dir(fullfile(TRAIN_OUT, '20*'));
if isempty(d)
    error('No model directory found in %s', TRAIN_OUT);
end
[~, sorted_idx] = sort({d.name});
LATEST_MODEL_DIR = fullfile(TRAIN_OUT, d(sorted_idx(end)).name);

ONNX_PATH = fullfile(LATEST_MODEL_DIR, 'residual_model.onnx');
STATS_PATH = fullfile(LATEST_MODEL_DIR, 'scaling_stats.json');

fprintf('Loading ONNX: %s\n', ONNX_PATH);

% Import the ONNX Model into the run output root to keep it clean
onnx_func_path = fullfile(RUN_OUT_ROOT, 'residual_net');
params_onnx = importONNXFunction(ONNX_PATH, onnx_func_path);
addpath(RUN_OUT_ROOT);

% Read JSON stats
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

% Read Validation Metrics JSON
METRICS_PATH = fullfile(LATEST_MODEL_DIR, 'validation_metrics.json');
if exist(METRICS_PATH, 'file')
    fid = fopen(METRICS_PATH);
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    metrics = jsondecode(str);
    learned_model.residual_variance = metrics.residual_variance;
    fprintf('  [OK] Loaded validation_metrics.json. Residual Variance = %f\n', learned_model.residual_variance);
else
    warning('[Learned Dynamics] validation_metrics.json not found at: %s.\n  Falling back to default residual_variance = %f. This may slightly over- or under-estimate SBC safety limits.', ...
        METRICS_PATH, 0.05^2);
    learned_model.residual_variance = 0.05^2; % Fallback variance
end

%% 4.1 Consistency Check (Physical Parameters)
PHYS_PATH = fullfile(LATEST_MODEL_DIR, 'physical_params.json');
if exist(PHYS_PATH, 'file')
    fprintf('--> Checking Physical Parameters Consistency...\n');
    fid = fopen(PHYS_PATH);
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    model_phys = jsondecode(str);
    
    % Get runtime defaults by a minimal run (1 step, no save)
    check_cfg = struct('case_id', 0, 'enable_plot', false, 'save_mat', false);
    check_cfg.params_overrides = struct('steps', 1);
    [check_traj, ~] = run_bb_reverse_once(check_cfg);
    runtime_params = check_traj.params;
    
    % Parameters to check
    params_to_check = {'alpha_v', 'alpha_x', 'dt', 'vmax', 'n'};
    epsilon = 1e-6;
    mismatch = false;
    
    for p = 1:numel(params_to_check)
        pname = params_to_check{p};
        if isfield(model_phys, pname) && isfield(runtime_params, pname)
            v_model = model_phys.(pname);
            v_runtime = runtime_params.(pname);
            if abs(v_model - v_runtime) > epsilon
                fprintf('  [ERROR] Mismatch in %s: Model=%f, Runtime=%f\n', pname, v_model, v_runtime);
                mismatch = true;
            else
                fprintf('  [OK] %s: %f\n', pname, v_model);
            end
        end
    end
    
    if mismatch
        error('Physical parameters mismatch between trained model and simulation! Please sync them in run_bb_reverse_once.m');
    end
else
    warning('physical_params.json not found in model directory. Skipping consistency check.');
end

%% 5. Load Case Manifest
output_root = fullfile(PROJECT_ROOT, 'traj', 'step3_collect');
files = dir(fullfile(output_root, '**', 'manifest.mat'));
if isempty(files)
    error('Manifest not found in %s', output_root);
end
[~, sorted_idx] = sort({files.name});
manifest_mat = fullfile(files(sorted_idx(end)).folder, files(sorted_idx(end)).name);

fprintf('--> Using Case Manifest: %s\n', manifest_mat);
S = load(manifest_mat);
manifest = S.manifest;
num_cases = numel(manifest);

% Initialize results manifest for data collection
results_manifest = repmat(struct( ...
    'case_id', 0, ...
    'seed', 0, ...
    'runtime_s', NaN, ...
    'status', 'skipped', ...
    'has_collision', false, ...
    'num_collisions', 0, ...
    'min_distance', NaN, ...
    'message', '', ...
    'output_file', ''), num_cases, 1);

collision_cases = [];
min_dist_overall = inf;

fprintf('\n开始执行 %d 个集成测试 Cases (使用 NN 替代 DM 预测)...\n', num_cases);
fprintf('Parallel Execution: %d\n', use_parallel);

if use_parallel
    parfor i = 1:num_cases
        if ~strcmp(manifest(i).status, 'ok')
            continue;
        end
        
        cfg = struct();
        cfg.case_id = manifest(i).case_id;
        cfg.seed = manifest(i).seed;
        cfg.enable_plot = false;
        cfg.save_mat = true;
        cfg.output_root = RUN_OUT_ROOT;
        
        % Core Injection of Neural Network into config overrides
        if isfield(manifest, 'params_overrides')
            cfg.params_overrides = manifest(i).params_overrides;
        else
            cfg.params_overrides = struct();
        end
        cfg.params_overrides.use_learned_dynamics = true;
        cfg.params_overrides.use_prsbc_filter = true;
        cfg.params_overrides.prsbc_use_nn = true;
        cfg.params_overrides.learned_model = learned_model;
        cfg.params_overrides.residual_variance = learned_model.residual_variance;
        cfg.params_overrides.control_noise_std = 0; % 禁用集成评估时的控制噪声
        cfg.params_overrides.explore_noise_std = 0; % 禁用集成评估时的探索噪声
        
        fprintf('Running Case %03d...\n', cfg.case_id);

        % Pre-allocate a local structure to store parfor results
        row = struct( ...
            'case_id', manifest(i).case_id, ...
            'seed', manifest(i).seed, ...
            'runtime_s', NaN, ...
            'status', 'ok', ...
            'has_collision', false, ...
            'num_collisions', 0, ...
            'min_distance', NaN, ...
            'message', '', ...
            'output_file', '');

        try
            [traj, run_info] = run_bb_reverse_once(cfg);
            row.runtime_s = run_info.runtime_s;
            row.output_file = run_info.output_file;
            
            % Check for Collisions Immediately
            x = traj.x; y = traj.y; n = traj.params.n;
            min_dist_case = inf;
            for t = 1:size(x,1)
                pos = [x(t,:); y(t,:)];
                for a1 = 1:n-1
                    for a2 = a1+1:n
                        d_val = norm(pos(:,a1)-pos(:,a2));
                        if d_val < min_dist_case, min_dist_case = d_val; end
                    end
                end
            end
            row.min_distance = min_dist_case;
            
            % Reduction Variables
            min_dist_overall = min(min_dist_overall, min_dist_case);
            
            [collision_info, has_collision] = check_collision(traj);
            row.has_collision = has_collision;
            row.num_collisions = numel(collision_info);
            
            if has_collision
                fprintf('Case %03d: [COLLISION] 碰撞次数: %d | 最小距离: %.4f\n', ...
                    cfg.case_id, row.num_collisions, min_dist_case);
                collision_cases = [collision_cases; manifest(i).case_id];
            else
                fprintf('Case %03d: [SAFE]      最小距离: %.4f\n', cfg.case_id, min_dist_case);
            end
            
        catch ME
            fprintf('Case %03d: [ERROR] 失败: %s\n', manifest(i).case_id, ME.message);
            row.status = 'failed';
            row.message = ME.message;
        end
        results_manifest(i) = row;
    end
else
    for i = 1:num_cases
        if ~strcmp(manifest(i).status, 'ok')
            continue;
        end
        
        cfg = struct();
        cfg.case_id = manifest(i).case_id;
        cfg.seed = manifest(i).seed;
        cfg.enable_plot = false;
        cfg.save_mat = true;
        cfg.output_root = RUN_OUT_ROOT;
        
        % Core Injection of Neural Network into config overrides
        if isfield(manifest, 'params_overrides')
            cfg.params_overrides = manifest(i).params_overrides;
        else
            cfg.params_overrides = struct();
        end
        cfg.params_overrides.use_learned_dynamics = true;
        cfg.params_overrides.use_prsbc_filter = true;
        cfg.params_overrides.prsbc_use_nn = true;
        cfg.params_overrides.learned_model = learned_model;
        cfg.params_overrides.residual_variance = learned_model.residual_variance;
        cfg.params_overrides.control_noise_std = 0; % 禁用集成评估时的控制噪声
        cfg.params_overrides.explore_noise_std = 0; % 禁用集成评估时的探索噪声
        
        fprintf('Running Case %03d... ', cfg.case_id);

        row = struct( ...
            'case_id', manifest(i).case_id, ...
            'seed', manifest(i).seed, ...
            'runtime_s', NaN, ...
            'status', 'ok', ...
            'has_collision', false, ...
            'num_collisions', 0, ...
            'min_distance', NaN, ...
            'message', '', ...
            'output_file', '');

        try
            [traj, run_info] = run_bb_reverse_once(cfg);
            row.runtime_s = run_info.runtime_s;
            row.output_file = run_info.output_file;
            
            % Check for Collisions Immediately
            x = traj.x; y = traj.y; n = traj.params.n;
            min_dist_case = inf;
            for t = 1:size(x,1)
                pos = [x(t,:); y(t,:)];
                for a1 = 1:n-1
                    for a2 = a1+1:n
                        d_val = norm(pos(:,a1)-pos(:,a2));
                        if d_val < min_dist_case, min_dist_case = d_val; end
                    end
                end
            end
            row.min_distance = min_dist_case;
            
            min_dist_overall = min(min_dist_overall, min_dist_case);
            
            [collision_info, has_collision] = check_collision(traj);
            row.has_collision = has_collision;
            row.num_collisions = numel(collision_info);
            
            if has_collision
                fprintf('[COLLISION] 碰撞次数: %d | 最小距离: %.4f\n', ...
                    row.num_collisions, min_dist_case);
                collision_cases = [collision_cases; manifest(i).case_id];
            else
                fprintf('[SAFE]      最小距离: %.4f\n', min_dist_case);
            end
            
        catch ME
            fprintf('[ERROR] 失败: %s\n', ME.message);
            row.status = 'failed';
            row.message = ME.message;
        end
        results_manifest(i) = row;
    end
end

%% 7. Final Report Summary
fprintf('\n========================================\n');
fprintf('集成测试分析总结 (Step4 Integrate):\n');
fprintf('检查总数: %d\n', num_cases);
fprintf('全局最小距离: %.4f\n', min_dist_overall);
fprintf('出现碰撞的 Case 数量: %d\n', numel(collision_cases));
if ~isempty(collision_cases)
    fprintf('碰撞 Case 编号: %s\n', num2str(collision_cases'));
else
    fprintf('恭喜！所有采用Learned Dynamics的轨迹均未发现碰撞。\n');
end
fprintf('========================================\n');

%% 8. Save Manifest
manifest = results_manifest; % Rename for compatibility
manifest_mat = fullfile(RUN_OUT_ROOT, 'manifest.mat');
save(manifest_mat, 'manifest');

manifest_csv = fullfile(RUN_OUT_ROOT, 'manifest.csv');
T = struct2table(manifest);
writetable(T, manifest_csv);

fprintf('Integration manifest saved to: %s\n', RUN_OUT_ROOT);
