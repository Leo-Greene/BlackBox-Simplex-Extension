function verify_deployment_v2(custom_val_mat, custom_model_dir)
% verify_deployment_v2: 使用验证集进行多步预测误差评估。
% 本脚本旨在通过与 test_deployment.m 使用完全相同的逻辑，
% 但切换数据集为 dataset_val.mat，来分析模型是否出现了泛化性能下降。

%% 1. 环境准备
PROJECT_ROOT = 'd:/Coding/Project/CPS/BlackBox-Simplex-Extension';
report_timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
VERIFY_OUT = fullfile(PROJECT_ROOT, 'experiment/test/out_verify', report_timestamp);
if ~exist(VERIFY_OUT, 'dir'), mkdir(VERIFY_OUT); end
addpath(VERIFY_OUT); 
addpath(genpath('.')); 
TRAIN_OUT = fullfile(PROJECT_ROOT, 'experiment/train/out');

% 自动寻找最新模型
if nargin < 2 || isempty(custom_model_dir)
    d = dir(fullfile(TRAIN_OUT, '20*'));
    [~, idx] = max([d.datenum]);
    LATEST_MODEL_DIR = fullfile(TRAIN_OUT, d(idx).name);
else
    LATEST_MODEL_DIR = custom_model_dir;
end

%% 2. 加载模型及统计量
ONNX_PATH = fullfile(LATEST_MODEL_DIR, 'residual_model.onnx');
STATS_PATH = fullfile(LATEST_MODEL_DIR, 'scaling_stats.json');

fprintf('--> Using Model: %s\n', LATEST_MODEL_DIR);
params_onnx = importONNXFunction(ONNX_PATH, fullfile(VERIFY_OUT, 'residual_net'));

fid = fopen(STATS_PATH);
stats_raw = jsondecode(char(fread(fid, inf)'));
fclose(fid);

stats.X_mean = reshape(double(stats_raw.X_mean), 1, []); 
stats.X_std  = reshape(double(stats_raw.X_std), 1, []);
stats.U_mean = reshape(double(stats_raw.U_mean), 1, []);
stats.U_std  = reshape(double(stats_raw.U_std), 1, []);
stats.R_mean = reshape(double(stats_raw.R_mean), 1, []);
stats.R_std  = reshape(double(stats_raw.R_std), 1, []);

%% 3. 数据加载逻辑 (强制指向 dataset_val.mat)
COLLECT_ROOT = fullfile(PROJECT_ROOT, 'traj', 'step3_collect');
if nargin > 0 && ~isempty(custom_val_mat)
    VAL_MAT = custom_val_mat;
else
    d2 = dir(fullfile(COLLECT_ROOT, '20*'));
    [~, sorted_idx] = sort({d2.name});
    VAL_MAT = fullfile(COLLECT_ROOT, d2(sorted_idx(end)).name, 'dataset_val.mat');
end

fprintf('--> Using VALIDATION Dataset: %s\n', VAL_MAT);

if exist(VAL_MAT, 'file')
    val_data = load(VAL_MAT); 
    X_val = val_data.split_ds.X;
    U_val = val_data.split_ds.U;

    % --- 核心修复：自动检测并校正转置 ---
    % 如果 X 的行数远大于理论状态维度(60)，说明它是 [Steps x State] 格式，需要转置
    expected_state_dim = 60; % 15 agents * 4
    if size(X_val, 1) > expected_state_dim && size(X_val, 2) == expected_state_dim
        X_val = X_val'; 
        U_val = U_val';
        fprintf('--> Auto-detected Data Transpose. Corrected to [State x Steps].\n');
    end
    
    % --- 轨迹边界检测 (核心修复版) ---
    if isfield(val_data.split_ds, 'case_id')
        case_id = val_data.split_ds.case_id;
        case_changes = find(case_id(1:end-1) ~= case_id(2:end));
        boundaries = [0, reshape(case_changes, 1, []), size(X_val, 2)];
    else
        pos_val = X_val(1:30, :);
        dist_sq = sum((pos_val(:, 2:end) - pos_val(:, 1:end-1)).^2, 1);
        boundaries = [0, reshape(find(dist_sq > 25.0), 1, []), size(X_val, 2)];
    end
    
    % --- 0. 提取原始 params ---
    parent_dir = fileparts(VAL_MAT);
    manifest_path = fullfile(parent_dir, 'manifest.mat');
    m_data = load(manifest_path);
    ref_case = m_data.cases(1);
    t_data = load(fullfile(parent_dir, sprintf('case_%03d', ref_case.case_id), sprintf('traj_case_%03d.mat', ref_case.case_id)));
    eval_params = t_data.traj.params;
    
    n_agents = eval_params.n;
    state_dim = 4 * n_agents;
    action_dim = 2 * n_agents;
    horizon = 10;
    num_samples = size(X_val, 2);
    results_table = struct('index', {}, 'error_norm', {}, 'avg_dist_err', {}, 'step_errors', {});

    fprintf('--> Starting Full Validation Scan...\n');
    
    for i = 1:num_samples - horizon
        if any(boundaries > i & boundaries < i + horizon), continue; end

        x_start = reshape(double(X_val(:, i)), 1, state_dim);
        u_seq = double(U_val(:, i:i+horizon-1))';
        
        % Rollout
        x_curr = x_start;
        current_step_errors = zeros(1, horizon);
        for t = 1:horizon
            [x_next, ~] = dynamics_learned(x_curr, u_seq(t, :), @residual_net, params_onnx, stats, eval_params);
            x_curr = x_next(1, :);
            x_true_t = double(X_val(:, i + t))';
            current_step_errors(t) = norm(x_curr - x_true_t);
        end
        
        diff = x_curr - double(X_val(:, i + horizon))';
        res.index = i;
        res.error_norm = norm(diff);
        res.avg_dist_err = mean(sqrt(diff(1:n_agents).^2 + diff(n_agents+1:2*n_agents).^2));
        res.step_errors = current_step_errors;
        results_table(end+1) = res;
    end

    %% 5. 生成报告 (对比用)
    report_file = fullfile(VERIFY_OUT, 'validation_error_report.txt');
    fid = fopen(report_file, 'w');
    fprintf(fid, '=== VALIDATION SET ERROR REPORT (For Generalization Check) ===\n');
    fprintf(fid, 'Model: %s\n', LATEST_MODEL_DIR);
    fprintf(fid, 'Total Samples: %d\n', length(results_table));
    fprintf(fid, 'Mean L2 Error: %.6f\n', mean([results_table.error_norm]));
    fprintf(fid, 'Max L2 Error:  %.6f\n', max([results_table.error_norm]));
    fclose(fid);
    
    fprintf('\n--> VALIDATION REPORT GENERATED: %s\n', report_file);
    fprintf('Mean Error on Validation: %.6f\n', mean([results_table.error_norm]));
end
end
