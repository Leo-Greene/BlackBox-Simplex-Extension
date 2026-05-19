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
PROJECT_ROOT = 'd:/Coding/Project/CPS/PrSBC-BlackBox-Simplex-Extension';

% 创建唯一的报告输出文件夹 (以当前时间戳命名)
report_timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
TEST_OUT = fullfile(PROJECT_ROOT, 'experiment/3_test_dynamics/out', report_timestamp);
if ~exist(TEST_OUT, 'dir'), mkdir(TEST_OUT); end

% 将输出路径添加到 MATLAB 搜索路径（优先级最高）
addpath(TEST_OUT, '-begin');
% 确保 parfor worker 能找到核心动力学函数
addpath(fullfile(PROJECT_ROOT, 'experiment/dynamics/learned_dynamics'));
% 避免根目录旧 residual_net.m 影子覆盖
root_residual = fullfile(PROJECT_ROOT, 'residual_net.m');
if exist(root_residual, 'file')
    warning('Removing shadowing residual_net.m at project root: %s', root_residual);
    delete(root_residual);
end
TRAIN_OUT = fullfile(PROJECT_ROOT, 'experiment/2_train/out');

% 自动寻找最新的模型文件夹 (如果没有手动指定 custom_model_dir)
if nargin < 2 || isempty(custom_model_dir)
    d = dir(fullfile(TRAIN_OUT, '20*'));
    if isempty(d)
        error('在 %s 目录下未找到任何模型文件夹。', TRAIN_OUT);
    end
    [~, idx_latest] = max([d.datenum]);
    LATEST_MODEL_DIR = fullfile(TRAIN_OUT, d(idx_latest).name);
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
    X_obs = val_data.split_ds.X;
    if isfield(val_data.split_ds, 'X_true')
        X_true = val_data.split_ds.X_true;
    else
        X_true = X_obs;
        warning('X_true missing; falling back to X (observed) for nominal/eval.');
    end
    U_val = val_data.split_ds.U;
    R_label = val_data.split_ds.R_label;

    % --- 4.1 数据维度校准 ---
    % 检测数据是 [Dim x Steps] 还是 [Steps x Dim] 并统一为 [Dim x Steps]
    expected_state_dim = 60; % 假设 15 agents * 4 (x,y,vx,vy)
    if size(X_obs, 2) == expected_state_dim
        X_obs = X_obs';
        X_true = X_true';
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
        boundaries = [0, reshape(case_changes, 1, []), size(X_obs, 2)];
    else
        % 如果缺失 case_id，通过位置大幅跳变来启发式检测边界
        pos_val = X_obs(1:30, :); 
        dist_sq = sum((pos_val(:, 2:end) - pos_val(:, 1:end-1)).^2, 1);
        boundaries = [0, reshape(find(dist_sq > 25.0), 1, []), size(X_obs, 2)];
    end
    
    % --- 4.3 物理参数提取 (重要：确保预测环境与数据一致) ---
    % 从原始轨迹数据中提取 dt, vmax, acc_scale 等参数，这些参数在 dynamics_learned 中被使用
    num_samples = size(X_obs, 2);
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
    fprintf('--> Starting Full Validation Scan (%d samples) using PARFOR...\n', num_samples);
    scan_step = 1; 
    
    % --- 预处理 parfor 需要的变量 ---
    has_case_id = isfield(val_data.split_ds, 'case_id');
    if has_case_id
        case_id_array = val_data.split_ds.case_id;
    else
        case_id_array = [];
    end
    is_u_transposed = (size(U_val, 1) == action_dim);
    
    % 预分配 cell 数组以供并行赋值
    results_cell = cell(num_samples - horizon, 1);
    
    parfor i = 1:scan_step:num_samples - horizon
        % 检查当前滑窗 [i, i+horizon] 是否跨越了轨迹边界
        valid_sample = true;
        if has_case_id
            if case_id_array(i) ~= case_id_array(i + horizon)
                valid_sample = false;
            end
        else
            if any(boundaries > i & boundaries <= i + horizon)
                valid_sample = false; 
            end
        end

        if ~valid_sample
            continue;
        end

        x_obs_start = reshape(double(X_obs(:, i)), 1, state_dim);
        x_nom_start = reshape(double(X_true(:, i)), 1, state_dim);
        
        % 获取真实的控制序列输入
        if is_u_transposed
            u_seq = double(U_val(:, i:i+horizon-1))'; 
        else
            u_seq = double(U_val(i:i+horizon-1, :));
        end
        
        % 执行多步推理
        x_nom_curr = x_nom_start;
        current_step_errors = zeros(1, horizon);
        for t = 1:horizon
            % 调用 Learned Dynamics (组合了 Nominal 模型与 Neural Residual)
            x_obs_curr = reshape(double(X_obs(:, i + t - 1)), 1, state_dim);
            [x_next_raw, ~] = dynamics_learned(x_nom_curr, x_obs_curr, u_seq(t, :), @residual_net, params_onnx, stats, eval_params);
            x_nom_curr = x_next_raw(1, :);
            
            % 计算每一时刻的累积 L2 误差
            x_true_t = double(X_true(:, i + t))'; 
            current_step_errors(t) = norm(x_nom_curr - x_true_t);
        end
        
        % 计算最终状态误差指标
        x_true_final = double(X_true(:, i + horizon))';
        diff = x_nom_curr - x_true_final;
        err_norm = norm(diff);
        
        % 解耦位置误差：计算所有 Agent 的平均距离偏移 (单位通常是米)
        dx = diff(1:n_agents);
        dy = diff(n_agents+1:2*n_agents);
        avg_dist = mean(sqrt(dx.^2 + dy.^2));

        % 缓存结果用于后续分析
        res = struct();
        res.index = i;
        res.error_norm = err_norm;
        res.avg_dist_err = avg_dist;
        res.step_errors = current_step_errors; 
        res.tag = 'unknown'; 
        res.case_id = 'N/A';
        results_cell{i} = res;
        
        if mod(i, 100) == 0, fprintf('Processed %d/%d...\n', i, num_samples); end
    end
    
    % 将 cell 合并为原来的 struct array
    valid_idx = ~cellfun(@isempty, results_cell);
    results_table = [results_cell{valid_idx}];

    %% =========================================================================
    %% 5. 生成分析报告 (TXT)
    %% =========================================================================
    % 依据多智能体解耦 (Multi-Agent Decoupling) 理念，采用平均距离误差作为最坏情况评判标准
    [max_dist_err, max_idx] = max([results_table.avg_dist_err]);
    worst_sample = results_table(max_idx);
    
    [~, min_idx] = min([results_table.avg_dist_err]);
    best_sample = results_table(min_idx);
    
    report_file = fullfile(TEST_OUT, 'error_analysis_report.txt');
    fid = fopen(report_file, 'w');
    fprintf(fid, '====================================================\n');
    fprintf(fid, '    BSA RESIDUAL MODEL ERROR ANALYSIS REPORT        \n');
    fprintf(fid, '    (Multi-Agent Decoupled Architecture 4->4)       \n');
    fprintf(fid, '         Generated on: %s\n', datestr(now));
    fprintf(fid, '         Model:        %s\n', LATEST_MODEL_DIR);
    fprintf(fid, '         Dataset:      %s\n', VAL_MAT);
    fprintf(fid, '====================================================\n\n');
    
    fprintf(fid, '1. OVERALL STATISTICS (10-Step Rollout):\n');
    fprintf(fid, '   - Total Samples Test:      %d\n', length(results_table));
    fprintf(fid, '   - Mean Avg Agent Dist Error:   %.6f m\n', mean([results_table.avg_dist_err]));
    fprintf(fid, '   - Median Avg Agent Dist Error: %.6f m\n', median([results_table.avg_dist_err]));
    fprintf(fid, '   - Max Avg Agent Dist Error:    %.6f m (Worst Case)\n', max_dist_err);
    fprintf(fid, '   - Mean Total 60D L2 Norm:      %.6f (Legacy Metric)\n\n', mean([results_table.error_norm]));
    
    fprintf(fid, '2. WORST CASE DETAIL:\n');
    fprintf(fid, '   - Sample Index in Test Set: %d\n', worst_sample.index);
    fprintf(fid, '   - Avg Agent Dist Error:    %.6f m\n', worst_sample.avg_dist_err);
    fprintf(fid, '   - Total 60D L2 Norm:       %.6f\n', worst_sample.error_norm);
    
    % 记录生成该数据集时的物理不一致性参数 (即 Ground Truth)
    if origin_found
        fprintf(fid, '   - Physical Discrepancy (Nominal Mismatch Parameters):\n');
        try
            if isfield(eval_params, 'alpha_v'),   fprintf(fid, '       * alpha_v:   %.3f\n', eval_params.alpha_v); end
            if isfield(eval_params, 'alpha_x'),   fprintf(fid, '       * alpha_x:   %.3f\n', eval_params.alpha_x); end
            if isfield(eval_params, 'dt'),        fprintf(fid, '       * dt:        %.3f\n', eval_params.dt); end
            if isfield(eval_params, 'n'),         fprintf(fid, '       * n_agents:  %d\n', eval_params.n); end
        catch
            fprintf(fid, '       * (Error displaying specific fields from eval_params)\n');
        end
    end
    fprintf(fid, '   - Suggestion: If error is high, check boundary collision frequency.\n\n');
    
    fprintf(fid, '3. ERROR DISTRIBUTION (Avg Agent Dist Error Range):\n');
    % 使用直方图分桶展示误差分布 (单位: 米)
    edges = [0, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0];
    counts = histcounts([results_table.avg_dist_err], edges);
    for k = 1:length(counts)
        fprintf(fid, '   - [%.2f - %.2f] m: %d samples\n', edges(k), edges(k+1), counts(k));
    end
    
    fclose(fid);
    fprintf('\n--> REPORT GENERATED: %s\n', report_file);
    fprintf('Worst Case Sample Index: %d, Max Avg Dist Error: %.6f m\n', worst_sample.index, max_dist_err);

    %% =========================================================================
    %% 6. 数据可视化分析
    %% =========================================================================
    figure('Name', 'Learned Dynamics Reliability Analysis', 'Visible', 'off'); 
    set(gcf, 'Position', [100, 100, 1200, 1000]); 
    
    % --- 6.1: 轨迹全时段误差分布 ---
    subplot(4, 1, 1);
    plot([results_table.index], [results_table.avg_dist_err], '-o', 'LineWidth', 1.5, 'Color', [0.85, 0.33, 0.1]);
    title(['10-Step Rollout Accuracy Analysis (Avg Distance Error) - ', report_timestamp], 'Interpreter', 'none');
    xlabel('Sample Start Index'); ylabel('Avg Agent Dist Error (m)');
    grid on;
    xticks(0:60:num_samples);
    xlim([0, num_samples]);
    % Add case boundary lines
    hold on;
    xline(0:60:num_samples, '--', 'Color', [0.5 0.5 0.5], 'Alpha', 0.3, 'HandleVisibility', 'off');

    % --- 子图 2: 预测时段内的平均速度 (Velocity vs. Error) ---
    avg_vel_horizon = NaN(num_samples, 1);
    for i = 1:num_samples - horizon
        % 计算该时段内所有 agent 的平均速度
        if size(X_true, 2) == state_dim
            vx_horizon = double(X_true(i:i+horizon-1, 2*n_agents+1:3*n_agents));
            vy_horizon = double(X_true(i:i+horizon-1, 3*n_agents+1:4*n_agents));
        else
            vx_horizon = double(X_true(2*n_agents+1:3*n_agents, i:i+horizon-1))';
            vy_horizon = double(X_true(3*n_agents+1:4*n_agents, i:i+horizon-1))';
        end
        % vx_horizon is [horizon x n_agents]
        vel_mags = sqrt(vx_horizon.^2 + vy_horizon.^2);
        avg_vel_horizon(i) = mean(vel_mags(:)); 
    end
    
    subplot(4, 1, 2);
    plot(1:num_samples, avg_vel_horizon, '-k', 'LineWidth', 1.2);
    ylabel('Avg Velocity (m/s)');
    hold on;
    yyaxis right
    plot([results_table.index], [results_table.avg_dist_err], '--r', 'LineWidth', 0.8);
    ylabel('Rollout Error (m)');
    grid on;
    xticks(0:60:num_samples);
    xlim([0, num_samples]);
    xline(0:60:num_samples, '--', 'Color', [0.5 0.5 0.5], 'Alpha', 0.3, 'HandleVisibility', 'off');
    title('Mean Velocity (10-step Horizon) vs. Error Trend');
    legend('Avg Speed (m/s)', 'Rollout Dist Error', 'Location', 'northwest');

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
    ylabel('Avg Acceleration (m/s^2)');
    hold on;
    yyaxis right
    plot([results_table.index], [results_table.avg_dist_err], '--r', 'LineWidth', 0.8);
    ylabel('Rollout Error (m)');
    grid on;
    xticks(0:60:num_samples);
    xlim([0, num_samples]);
    xline(0:60:num_samples, '--', 'Color', [0.5 0.5 0.5], 'Alpha', 0.3, 'HandleVisibility', 'off');
    title('Future Action Intensity (Mean Acc over 10 steps) vs. Error');
    legend('Avg Future Acc', 'Rollout Dist Error', 'Location', 'northwest');

    % --- 子图 4: 位置特征 (离中心的距离) ---
    dist_to_center = zeros(num_samples, 1);
    for k = 1:num_samples
        if size(X_true, 2) == state_dim
            px = double(X_true(k, 1:n_agents));
            py = double(X_true(k, n_agents+1:2*n_agents));
        else
            px = double(X_true(1:n_agents, k));
            py = double(X_true(n_agents+1:2*n_agents, k));
        end
        dist_to_center(k) = mean(sqrt(px.^2 + py.^2));
    end
    subplot(4, 1, 4);
    plot(1:num_samples, dist_to_center, '-b', 'LineWidth', 1.2);
    grid on;
    xticks(0:60:num_samples);
    xlim([0, num_samples]);
    hold on;
    xline(0:60:num_samples, '--', 'Color', [0.5 0.5 0.5], 'Alpha', 0.3, 'HandleVisibility', 'off');
    title('Scenario Progress (Avg Dist to Origin)');
    xlabel('Sample Index (Time Step)');
    ylabel('Mean Distance (m)');
    
    % 保存图片到日期文件夹
    plot_file = fullfile(TEST_OUT, 'causality_analysis_plot.png');
    saveas(gcf, plot_file);
    fprintf('--> CAUSALITY ANALYSIS PLOT SAVED: %s\n', plot_file);
    close(gcf);

    %% 6.2 绘图：误差随预测步数 (1-10步) 的演化图 (Error Propagation)
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

    %% 6.3 绘图：最差情况 (Worst Case) 真实轨迹 vs 预测轨迹对比
    % 重新模拟最差情况，便于获取完整轨迹
    idx_worst = worst_sample.index;
    x_start_worst = X_true(:, idx_worst);
    if size(U_val, 1) == action_dim
        u_seq_worst = double(U_val(:, idx_worst:idx_worst+horizon-1))'; 
    else
        u_seq_worst = double(U_val(idx_worst:idx_worst+horizon-1, :));
    end

    x_nom_curr = reshape(x_start_worst, 1, state_dim);
    pred_traj = zeros(horizon + 1, state_dim);
    pred_traj(1, :) = x_nom_curr;

    for t = 1:horizon
        x_obs_curr = reshape(double(X_obs(:, idx_worst + t - 1)), 1, state_dim);
        [x_next_raw, ~] = dynamics_learned(x_nom_curr, x_obs_curr, u_seq_worst(t, :), @residual_net, params_onnx, stats, eval_params);
        x_nom_curr = x_next_raw(1, :);
        pred_traj(t+1, :) = x_nom_curr;
    end

    true_traj = double(X_true(:, idx_worst:idx_worst+horizon))'; % [horizon+1 x state_dim]

    figure('Name', 'Worst Case Trajectory Comparison', 'Visible', 'off');
    set(gcf, 'Position', [100, 100, 800, 800]);
    hold on;

    h_true = []; h_pred = []; h_start = []; h_end_t = []; h_end_p = [];

    % 遍历绘制每一个 Agent 的位移轨迹
    for j = 1:n_agents
        p1 = plot(true_traj(:, j), true_traj(:, n_agents+j), '-b', 'LineWidth', 1.5, 'Marker', '.');
        p2 = plot(pred_traj(:, j), pred_traj(:, n_agents+j), '--r', 'LineWidth', 1.5, 'Marker', 'x');
        
        % 标记起点、终点位置
        p3 = plot(true_traj(1, j), true_traj(1, n_agents+j), 'og', 'MarkerFaceColor', 'g', 'MarkerSize', 6);
        p4 = plot(true_traj(end, j), true_traj(end, n_agents+j), 'sb', 'MarkerFaceColor', 'b', 'MarkerSize', 6);
        p5 = plot(pred_traj(end, j), pred_traj(end, n_agents+j), '^r', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
        
        % 收集供图例使用的线把柄
        if j == 1
            h_true = p1; h_pred = p2; h_start = p3; h_end_t = p4; h_end_p = p5;
        end
    end

    grid on; axis equal;
    title(sprintf('Worst Case Trajectory Comparison (Index: %d, Max L2 Error: %.2f)', idx_worst, worst_sample.error_norm));
    xlabel('X Position (m)'); ylabel('Y Position (m)');
    
    % 添加图例说明
    legend([h_true, h_pred, h_start, h_end_t, h_end_p], ...
           {'True Trajectory', 'Predicted Trajectory', 'Start Position', 'True End Position', 'Predicted End Position'}, ...
           'Location', 'best');

    plot_file = fullfile(TEST_OUT, 'worst_case_comparison_plot.png');
    saveas(gcf, plot_file);
    fprintf('--> WORST CASE COMPARISON PLOT SAVED: %s\n', plot_file);
    close(gcf);

    %% 6.4 绘图：最好情况 (Best Case) 真实轨迹 vs 预测轨迹对比
    % 重新模拟最好情况 (误差最小的样本)
    idx_best = best_sample.index;
    x_start_best = X_true(:, idx_best);
    if size(U_val, 1) == action_dim
        u_seq_best = double(U_val(:, idx_best:idx_best+horizon-1))'; 
    else
        u_seq_best = double(U_val(idx_best:idx_best+horizon-1, :));
    end

    x_nom_curr = reshape(x_start_best, 1, state_dim);
    pred_traj_best = zeros(horizon + 1, state_dim);
    pred_traj_best(1, :) = x_nom_curr;

    for t = 1:horizon
        x_obs_curr = reshape(double(X_obs(:, idx_best + t - 1)), 1, state_dim);
        [x_next_raw, ~] = dynamics_learned(x_nom_curr, x_obs_curr, u_seq_best(t, :), @residual_net, params_onnx, stats, eval_params);
        x_nom_curr = x_next_raw(1, :);
        pred_traj_best(t+1, :) = x_nom_curr;
    end

    true_traj_best = double(X_true(:, idx_best:idx_best+horizon))'; % [horizon+1 x state_dim]

    figure('Name', 'Best Case Trajectory Comparison', 'Visible', 'off');
    set(gcf, 'Position', [200, 200, 800, 800]);
    hold on;

    h_true = []; h_pred = []; h_start = []; h_end_t = []; h_end_p = [];

    % 遍历绘制每一个 Agent 的位移轨迹
    for j = 1:n_agents
        p1 = plot(true_traj_best(:, j), true_traj_best(:, n_agents+j), '-b', 'LineWidth', 1.5, 'Marker', '.');
        p2 = plot(pred_traj_best(:, j), pred_traj_best(:, n_agents+j), '--r', 'LineWidth', 1.5, 'Marker', 'x');
        
        % 标记起点、终点位置
        p3 = plot(true_traj_best(1, j), true_traj_best(1, n_agents+j), 'og', 'MarkerFaceColor', 'g', 'MarkerSize', 6);
        p4 = plot(true_traj_best(end, j), true_traj_best(end, n_agents+j), 'sb', 'MarkerFaceColor', 'b', 'MarkerSize', 6);
        p5 = plot(pred_traj_best(end, j), pred_traj_best(end, n_agents+j), '^r', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
        
        % 收集供图例使用的线把柄
        if j == 1
            h_true = p1; h_pred = p2; h_start = p3; h_end_t = p4; h_end_p = p5;
        end
    end

    grid on; axis equal;
    title(sprintf('Best Case Trajectory Comparison (Index: %d, Min L2 Error: %.2f)', idx_best, best_sample.error_norm));
    xlabel('X Position (m)'); ylabel('Y Position (m)');
    
    % 添加图例说明
    legend([h_true, h_pred, h_start, h_end_t, h_end_p], ...
           {'True Trajectory', 'Predicted Trajectory', 'Start Position', 'True End Position', 'Predicted End Position'}, ...
           'Location', 'best');

    plot_file_best = fullfile(TEST_OUT, 'best_case_comparison_plot.png');
    saveas(gcf, plot_file_best);
    fprintf('--> BEST CASE COMPARISON PLOT SAVED: %s\n', plot_file_best);
    close(gcf);

else
    warning('Validation dataset not found.');
end

fprintf('\nStep 4 Part A & B: Integration & Rollout Finished.\n');
