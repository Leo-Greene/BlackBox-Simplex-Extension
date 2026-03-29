function test_deployment(custom_test_mat, custom_model_dir)
% Step 4: Integrated Test Script
% 本脚本用于在 MATLAB 环境下评估 Learned Dynamics 的多步预测效果。
%
% 功能描述:
%   1. 自动加载最新的训练模型 (ONNX) 和归一化参数 (JSON)。
%   2. 自动加载最新的多 Agent 轨迹数据集。
%   3. 进行多步 Rollout 预测 (默认 10 步)，并通过真实值对比计算误差。
%   4. 生成详细的分析报告和可视化图表。
%
% 用法:
%   test_deployment                         (自动加载最新的测试数据集和最新的模型)
%   test_deployment(data_path)               (指定数据集，模型用最新的)
%   test_deployment([], model_path)           (数据集用最新的，指定模型目录名/路径)
%   test_deployment(data_path, model_path)    (同时手动指定)

%% =========================================================================
%% 1. 环境准备与路径配置
%% =========================================================================
PROJECT_ROOT = 'd:/Coding/Project/CPS/BlackBox-Simplex-Extension';

% 创建唯一的报告输出文件夹 (以当前时间戳命名)
report_timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
TEST_OUT = fullfile(PROJECT_ROOT, 'experiment/test/out', report_timestamp);
if ~exist(TEST_OUT, 'dir'), mkdir(TEST_OUT); end

% 将输出路径和子目录添加到 MATLAB 搜索路径
addpath(TEST_OUT); 
addpath(genpath('.')); 
TRAIN_OUT = fullfile(PROJECT_ROOT, 'experiment/train/out');

% 自动寻找最新的模型文件夹 (如果没有手动指定 custom_model_dir)
if nargin < 2 || isempty(custom_model_dir)
    d = dir(fullfile(TRAIN_OUT, '20*'));
    if isempty(d)
        error('在 %s 目录下未找到任何模型文件夹。', TRAIN_OUT);
    end
    [~, idx] = max([d.datenum]);
    LATEST_MODEL_DIR = fullfile(TRAIN_OUT, d(idx).name);
else
    % 检查指定的路径是否为绝对路径或相对于 TRAIN_OUT 的路径
    if isfolder(custom_model_dir)
        LATEST_MODEL_DIR = custom_model_dir;
    else
        LATEST_MODEL_DIR = fullfile(TRAIN_OUT, custom_model_dir);
        if ~exist(LATEST_MODEL_DIR, 'dir')
            error('指定的模型目录不存在: %s', custom_model_dir);
        end
    end
end

%% =========================================================================
%% 2. 模型加载与归一化参数解析
%% =========================================================================
ONNX_PATH = fullfile(LATEST_MODEL_DIR, 'residual_model.onnx');
STATS_PATH = fullfile(LATEST_MODEL_DIR, 'scaling_stats.json');

if ~exist(ONNX_PATH, 'file'), error('ONNX 模型不存在: %s', ONNX_PATH); end
if ~exist(STATS_PATH, 'file'), error('统计量文件不存在: %s', STATS_PATH); end

fprintf('--> Using Model: %s\n', LATEST_MODEL_DIR);
fprintf('--> Loading ONNX: %s\n', ONNX_PATH);

% 将 ONNX 模型导入为 MATLAB 函数
% 注意：params_onnx 包含模型权重，必须在调用生成函数时传入
params_onnx = importONNXFunction(ONNX_PATH, fullfile(TEST_OUT, 'residual_net'));

% 读取 Python 训练端导出的归一化统计量 (JSON 格式)
fid = fopen(STATS_PATH);
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
stats_raw = jsondecode(str);

% 核心处理：确保所有统计量均为 1xN 行向量，方便矩阵运算
stats.X_mean = reshape(double(stats_raw.X_mean), 1, []); 
stats.X_std  = reshape(double(stats_raw.X_std), 1, []);
stats.U_mean = reshape(double(stats_raw.U_mean), 1, []);
stats.U_std  = reshape(double(stats_raw.U_std), 1, []);
stats.R_mean = reshape(double(stats_raw.R_mean), 1, []);
stats.R_std  = reshape(double(stats_raw.R_std), 1, []);

%% =========================================================================
%% 3. 数据集定位逻辑
%% =========================================================================
COLLECT_ROOT = fullfile(PROJECT_ROOT, 'traj', 'step3_collect');
if nargin > 0 && ~isempty(custom_test_mat) % 修复变量名不一致
    if isfolder(custom_test_mat)
        VAL_MAT = fullfile(custom_test_mat, 'dataset_test.mat');
        if ~exist(VAL_MAT, 'file')
            error('在目录 %s 下未找到 dataset_test.mat。', custom_test_mat);
        end
    elseif exist(custom_test_mat, 'file')
        VAL_MAT = custom_test_mat;
    else
        VAL_MAT = fullfile(COLLECT_ROOT, custom_test_mat);
        if ~exist(VAL_MAT, 'file')
            error('指定的数据集文件或目录不存在: %s', custom_test_mat);
        end
    end
else
    % 默认选择最新的采集批次下的测试集
    d2 = dir(fullfile(COLLECT_ROOT, '20*'));
    if isempty(d2)
        error('在 %s 目录下未找到任何采集数据。', COLLECT_ROOT);
    end
    [~, sorted_idx] = sort({d2.name});
    idx2 = sorted_idx(end); 
    VAL_MAT = fullfile(COLLECT_ROOT, d2(idx2).name, 'dataset_test.mat');
end

fprintf('--> Using Dataset: %s\n', VAL_MAT);
fprintf('--> Evaluating Rollout Consistency...\n');

%% =========================================================================
%% 4. 数据预处理与 Rollout 评估
%% =========================================================================
if exist(VAL_MAT, 'file')
    % 加载测试集数据
    val_data = load(VAL_MAT); 
    X_val = val_data.split_ds.X;
    U_val = val_data.split_ds.U;
    R_label = val_data.split_ds.R_label;

    % --- 4.1 数据维度校准 ---
    % 检测数据是 [Dim x Steps] 还是 [Steps x Dim] 并统一为 [Dim x Steps]
    expected_state_dim = 60; % 假设 15 agents * 4 (x,y,vx,vy)
    if size(X_val, 1) > expected_state_dim && size(X_val, 2) == expected_state_dim
        X_val = X_val'; 
        U_val = U_val';
        fprintf('--> Auto-detected Data Transpose. Corrected to [State x Steps].\n');
    end
    
    % --- 4.2 轨迹边界检测 ---
    % 评估 Rollout 时不能跨越不同的轨迹(Case)，需识别轨迹跳变点
    fprintf('--> Identifying trajectory boundaries...\n');
    if isfield(val_data.split_ds, 'case_id')
        case_id = val_data.split_ds.case_id;
        % 找到 case_id 变化的位置
        case_changes = find(case_id(1:end-1) ~= case_id(2:end));
        boundaries = [0, reshape(case_changes, 1, []), size(X_val, 2)];
    else
        % 如果缺失 case_id，通过位置大幅跳变来启发式检测边界
        pos_val = X_val(1:30, :); 
        dist_sq = sum((pos_val(:, 2:end) - pos_val(:, 1:end-1)).^2, 1);
        boundaries = [0, reshape(find(dist_sq > 25.0), 1, []), size(X_val, 2)];
    end
    
    % --- 4.3 物理参数提取 (重要：确保预测环境与数据一致) ---
    % 从原始轨迹数据中提取 dt, vmax, acc_scale 等参数，这些参数在 dynamics_learned 中被使用
    num_samples = size(X_val, 2);
    horizon = 10; % 多步预测步长
    results_table = struct('index', {}, 'error_norm', {}, 'avg_dist_err', {}, 'step_errors', {}, 'tag', {}, 'case_id', {});
    
    parent_dir = fileparts(VAL_MAT);
    manifest_path = fullfile(parent_dir, 'manifest.mat');
    origin_found = false;
    
    if exist(manifest_path, 'file')
        m_data = load(manifest_path);
        if isfield(m_data, 'cases')
            ref_case = m_data.cases(1);
            try
                [m_folder, ~, ~] = fileparts(manifest_path);
                sample_traj_file = fullfile(m_folder, sprintf('case_%03d', ref_case.case_id), sprintf('traj_case_%03d.mat', ref_case.case_id));
                
                fprintf('--> Debug: Extracting params from %s\n', sample_traj_file);
                
                if exist(sample_traj_file, 'file')
                    t_data = load(sample_traj_file);
                    if isfield(t_data, 'traj') && isfield(t_data.traj, 'params')
                        % 获取采集时的完整物理配置
                        eval_params = t_data.traj.params; 
                        origin_found = true;
                    end
                end
            catch ME
                fprintf('--> Debug: Error during param extraction: %s\n', ME.message);
                origin_found = false;
            end
        end
    end

    if ~origin_found
        error('无法从数据集 %s 中提取原始 params。请确保 manifest.mat 和 case 文件夹完整。', VAL_MAT);
    end

    n_agents = eval_params.n;
    state_dim = 4 * n_agents;
    action_dim = 2 * n_agents;

    % --- 4.4 核心循环：多步 Rollout 评估 ---
    fprintf('--> Starting Full Validation Scan (%d samples)...\n', num_samples);
    scan_step = 1; 
    
    for i = 1:scan_step:num_samples - horizon
        % 检查当前滑窗 [i, i+horizon] 是否跨越了轨迹边界
        if isfield(val_data.split_ds, 'case_id')
            if val_data.split_ds.case_id(i) ~= val_data.split_ds.case_id(i + horizon)
                continue;
            end
        else
            if any(boundaries > i & boundaries <= i + horizon)
                continue; 
            end
        end

        current_x = double(X_val(:, i));
        x_start = reshape(current_x, 1, state_dim);
        
        % 获取真实的控制序列输入
        if size(U_val, 1) == action_dim
            u_seq = double(U_val(:, i:i+horizon-1))'; 
        else
            u_seq = double(U_val(i:i+horizon-1, :));
        end
        
        % 执行多步推理
        x_curr = x_start;
        current_step_errors = zeros(1, horizon);
        for t = 1:horizon
            % 调用 Learned Dynamics (组合了 Nominal 模型与 Neural Residual)
            [x_next_raw, ~] = dynamics_learned(x_curr, u_seq(t, :), @residual_net, params_onnx, stats, eval_params);
            x_curr = x_next_raw(1, :);
            
            % 计算每一时刻的累积 L2 误差
            x_true_t = double(X_val(:, i + t))'; 
            current_step_errors(t) = norm(x_curr - x_true_t);
        end
        
        % 计算最终状态误差指标
        x_true_final = double(X_val(:, i + horizon))';
        diff = x_curr - x_true_final;
        err_norm = norm(diff);
        
        % 解耦位置误差：计算所有 Agent 的平均距离偏移 (单位通常是米)
        dx = diff(1:n_agents);
        dy = diff(n_agents+1:2*n_agents);
        avg_dist = mean(sqrt(dx.^2 + dy.^2));

        % 缓存结果用于后续分析
        res.index = i;
        res.error_norm = err_norm;
        res.avg_dist_err = avg_dist;
        res.step_errors = current_step_errors; 
        res.tag = 'unknown'; 
        res.case_id = 'N/A';
        results_table(end+1) = res;
        
        if mod(i, 100) == 0, fprintf('Processed %d/%d...\n', i, num_samples); end
    end

    %% =========================================================================
    %% 5. 生成分析报告 (TXT)
    %% =========================================================================
    [max_err, max_idx] = max([results_table.error_norm]);
    worst_sample = results_table(max_idx);
    
    report_file = fullfile(TEST_OUT, 'error_analysis_report.txt');
    fid = fopen(report_file, 'w');
    fprintf(fid, '====================================================\n');
    fprintf(fid, '         RESIDUAL MODEL ERROR ANALYSIS REPORT       \n');
    fprintf(fid, '         Generated on: %s\n', datestr(now));
    fprintf(fid, '         Model:        %s\n', LATEST_MODEL_DIR);
    fprintf(fid, '         Dataset:      %s\n', VAL_MAT);
    fprintf(fid, '====================================================\n\n');
    
    fprintf(fid, '1. OVERALL STATISTICS (10-Step Rollout):\n');
    fprintf(fid, '   - Total Samples Test:      %d\n', length(results_table));
    fprintf(fid, '   - Mean L2 Norm Error:      %.6f\n', mean([results_table.error_norm]));
    fprintf(fid, '   - Median L2 Norm Error:    %.6f\n', median([results_table.error_norm]));
    fprintf(fid, '   - Max L2 Norm Error:       %.6f (Worst Case)\n\n', max_err);
    
    fprintf(fid, '2. WORST CASE DETAIL:\n');
    fprintf(fid, '   - Sample Index in Test Set: %d\n', worst_sample.index);
    fprintf(fid, '   - L2 Norm Error:           %.6f\n', worst_sample.error_norm);
    fprintf(fid, '   - Avg Agent Dist Error:    %.6f m\n', worst_sample.avg_dist_err);
    
    % 记录生成该数据集时的物理不一致性参数 (即 Ground Truth)
    if origin_found
        fprintf(fid, '   - Physical Discrepancy (Ground Truth Settings):\n');
        try
            if isfield(eval_params, 'acc_scale'), fprintf(fid, '       * acc_scale: %.3f (Nominal: 1.000)\n', eval_params.acc_scale); end
            if isfield(eval_params, 'damping'),   fprintf(fid, '       * damping:   %.3f\n', eval_params.damping); end
            if isfield(eval_params, 'vmax'),      fprintf(fid, '       * vmax:      %.3f\n', eval_params.vmax); end
            if isfield(eval_params, 'dt'),        fprintf(fid, '       * dt:        %.3f\n', eval_params.dt); end
            if isfield(eval_params, 'n'),         fprintf(fid, '       * n_agents:  %d\n', eval_params.n); end
        catch
            fprintf(fid, '       * (Error displaying specific fields from eval_params)\n');
        end
    end
    fprintf(fid, '   - Suggestion: If error is high, check boundary collision frequency.\n\n');
    
    fprintf(fid, '3. ERROR DISTRIBUTION (L2 Norm Range):\n');
    % 使用直方图分桶展示误差分布
    edges = [0, 0.5, 1, 2, 5, 10, 50];
    counts = histcounts([results_table.error_norm], edges);
    for k = 1:length(counts)
        fprintf(fid, '   - [%.1f - %.1f]: %d samples\n', edges(k), edges(k+1), counts(k));
    end
    
    fclose(fid);
    fprintf('\n--> REPORT GENERATED: %s\n', report_file);
    fprintf('Worst Case Sample Index: %d, Final Error (L2): %.6f\n', worst_sample.index, max_err);

    %% =========================================================================
    %% 6. 数据可视化分析
    %% =========================================================================
    figure('Name', 'Learned Dynamics Reliability Analysis', 'Visible', 'off'); 
    set(gcf, 'Position', [100, 100, 1200, 1000]); 
    
    % --- 6.1: 轨迹全时段误差分布 ---
    subplot(4, 1, 1);
    plot([results_table.index], [results_table.error_norm], '-o', 'LineWidth', 1.5, 'Color', [0.85, 0.33, 0.1]);
    title('10-Step Rollout Cumulative Error (Full Dataset Scan)');
    xlabel('Sample Start Index'); ylabel('L2 Norm Error');
    grid on;
    title(['10-Step Rollout Accuracy Analysis - ', report_timestamp], 'Interpreter', 'none');
    ylabel('L2 Norm Error');

    % --- 子图 2: 当前时刻的平均速度 (Correlation) ---
    avg_vel_mag = zeros(num_samples, 1);
    for k = 1:num_samples
        if size(X_val, 2) == state_dim
            vx_row = double(X_val(k, 2*n_agents+1:3*n_agents));
            vy_row = double(X_val(k, 3*n_agents+1:4*n_agents));
        else
            vx_row = double(X_val(2*n_agents+1:3*n_agents, k));
            vy_row = double(X_val(3*n_agents+1:4*n_agents, k));
        end
        avg_vel_mag(k) = mean(sqrt(vx_row.^2 + vy_row.^2));
    end
    subplot(4, 1, 2);
    plot(1:num_samples, avg_vel_mag, '-k', 'LineWidth', 1.2);
    hold on;
    yyaxis right
    plot([results_table.index], [results_table.error_norm], '--r', 'LineWidth', 0.8);
    ylabel('Rollout Error');
    grid on;
    title('Current Velocity vs. Error Trend');
    legend('Avg Speed (m/s)', 'Rollout Error', 'Location', 'northwest');

    % --- 子图 3: 未来10步的累计加速度强度 (Future Intensity) ---
    % 这个指标反映了预测任务的“剧烈程度”
    avg_acc_future = NaN(num_samples, 1);
    for i = 1:num_samples - horizon
        if size(U_val, 1) == action_dim
            u_future = double(U_val(:, i:i+horizon-1)); 
        else
            u_future = double(U_val(i:i+horizon-1, :))'; 
        end
        % 计算合加速度 norm，然后求 n_agents 个 agent 和 10 个步端的平均值
        acc_norms = sqrt(sum(u_future.^2, 1)); % 每一步总体的加速度瞬时值
        avg_acc_future(i) = mean(acc_norms);
    end

    subplot(4, 1, 3);
    plot(1:num_samples, avg_acc_future, '-m', 'LineWidth', 1.2);
    hold on;
    yyaxis right
    plot([results_table.index], [results_table.error_norm], '--r', 'LineWidth', 0.8);
    ylabel('Rollout Error');
    grid on;
    title('Future Action Intensity (Mean Acc over 10 steps) vs. Error');
    legend('Avg Future Acc', 'Rollout Error', 'Location', 'northwest');

    % --- 子图 4: 位置特征 (离中心的距离) ---
    dist_to_center = zeros(num_samples, 1);
    for k = 1:num_samples
        if size(X_val, 2) == state_dim
            px = double(X_val(k, 1:n_agents));
            py = double(X_val(k, n_agents+1:2*n_agents));
        else
            px = double(X_val(1:n_agents, k));
            py = double(X_val(n_agents+1:2*n_agents, k));
        end
        dist_to_center(k) = mean(sqrt(px.^2 + py.^2));
    end
    subplot(4, 1, 4);
    plot(1:num_samples, dist_to_center, '-b', 'LineWidth', 1.2);
    grid on;
    title('Scenario Progress (Avg Dist to Origin)');
    xlabel('Sample Index (Time Step)');
    ylabel('Mean Distance (m)');
    
    % 保存图片到日期文件夹
    plot_file = fullfile(TEST_OUT, 'causality_analysis_plot.png');
    saveas(gcf, plot_file);
    fprintf('--> CAUSALITY ANALYSIS PLOT SAVED: %s\n', plot_file);
    close(gcf);

    %% 10. 绘图：误差随预测步数 (1-10步) 的演化图 (Error Propagation)
    figure('Name', 'Multi-step Error Propagation', 'Visible', 'off');
    set(gcf, 'Position', [100, 100, 1000, 600]);
    
    % 将所有起始点的步进误差矩阵化 [样本数 x 10]
    all_step_errors = vertcat(results_table.step_errors);
    
    % 画出平均误差演化线
    mean_propagation = mean(all_step_errors, 1);
    std_propagation = std(all_step_errors, 1);
    
    hold on;
    % 画阴影区域 (置信区间)
    x_steps = 1:horizon;
    fill([x_steps, fliplr(x_steps)], ...
         [mean_propagation + std_propagation, fliplr(mean_propagation - std_propagation)], ...
         [0.8 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    
    % 画平均线
    plot(x_steps, mean_propagation, '-b', 'LineWidth', 2.5, 'Marker', 'o');
    
    % 额外：画出 Worst Case 的那条线进行对比
    worst_step_line = results_table(max_idx).step_errors;
    plot(x_steps, worst_step_line, '--r', 'LineWidth', 2, 'Marker', 'x');
    
    grid on;
    xticks(1:horizon);
    xlabel('Prediction Step (Horizon Index)');
    ylabel('L2 Norm Error (Accumulated)');
    title(['Error Propagation over 10-step Horizon - ', report_timestamp], 'Interpreter', 'none');
    legend('Std Dev (Spread)', 'Mean Error Propagation', 'Worst Case Trace', 'Location', 'northwest');
    
    % 保存图片到日期文件夹
    prop_plot_file = fullfile(TEST_OUT, 'error_propagation_plot.png');
    saveas(gcf, prop_plot_file);
    fprintf('--> ERROR PROPAGATION PLOT SAVED: %s\n', prop_plot_file);
    close(gcf);

else
    warning('Validation dataset not found.');
end

fprintf('\nStep 4 Part A & B: Integration & Rollout Finished.\n');
