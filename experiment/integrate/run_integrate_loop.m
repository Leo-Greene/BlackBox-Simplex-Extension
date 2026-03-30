% run_integrate_loop.m
% Integrated Evaluation of Learned Dynamics Model (Decision Module Multi-step)
clc;
clear;
close all;

%% 1. Path Setup
PROJECT_ROOT = fullfile(fileparts(mfilename('fullpath')), '..', '..');
addpath(genpath(fullfile(PROJECT_ROOT, 'decision_module')));
addpath(fullfile(PROJECT_ROOT, 'common'));
addpath(fullfile(PROJECT_ROOT, 'experiment', 'other')); % for check_collision
addpath(genpath(fullfile(PROJECT_ROOT, 'experiment', 'dynamics', 'learned_dynamics')));
addpath(fullfile(PROJECT_ROOT, 'experiment', 'data_collection')); % run_bb_reverse_once

%% 2. Setup Integration Run Environment
report_timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
RUN_OUT_ROOT = fullfile(PROJECT_ROOT, 'traj', 'step4_integrate', report_timestamp);
if ~exist(RUN_OUT_ROOT, 'dir'), mkdir(RUN_OUT_ROOT); end

%% 3. Load Neural Network Model & Stats
fprintf('--> Initializing Learned Dynamics Module...\n');
TRAIN_OUT = fullfile(PROJECT_ROOT, 'experiment', 'train', 'out');
d = dir(fullfile(TRAIN_OUT, '20*'));
if isempty(d)
    error('No model directory found in %s', TRAIN_OUT);
end
[~, idx] = max([d.datenum]);
LATEST_MODEL_DIR = fullfile(TRAIN_OUT, d(idx).name);

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

%% 4. Load Case Manifest
output_root = fullfile(PROJECT_ROOT, 'traj', 'step3_collect');
files = dir(fullfile(output_root, '**', 'manifest.mat'));
if isempty(files)
    error('Manifest not found in %s', output_root);
end
[~, idx] = max([files.datenum]);
manifest_mat = fullfile(files(idx).folder, files(idx).name);

fprintf('--> Using Case Manifest: %s\n', manifest_mat);
S = load(manifest_mat);
manifest = S.manifest;
num_cases = numel(manifest);

%% 5. Evaluate Loop
collision_cases = [];
total_steps_checked = 0;
min_dist_overall = inf;

fprintf('\n开始执行 %d 个集成测试 Cases (使用 NN 替代 DM 预测)...\n', num_cases);

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
    cfg.params_overrides.learned_model = learned_model;
    
    fprintf('Running Case %03d... ', cfg.case_id);
    
    try
        [traj, run_info] = run_bb_reverse_once(cfg);
        
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
        if min_dist_case < min_dist_overall, min_dist_overall = min_dist_case; end
        
        [collision_info, has_collision] = check_collision(traj);
        
        if has_collision
            fprintf('[COLLISION] 碰撞次数: %d | 最小距离: %.4f\n', ...
                numel(collision_info), min_dist_case);
            collision_cases = [collision_cases; manifest(i).case_id];
        else
            fprintf('[SAFE]      最小距离: %.4f\n', min_dist_case);
        end
        
    catch ME
        fprintf('[ERROR] 失败: %s\n', ME.message);
    end
end

%% 6. Final Report Summary
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
