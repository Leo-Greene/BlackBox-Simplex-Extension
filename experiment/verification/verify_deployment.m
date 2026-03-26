function verify_deployment(custom_val_mat, custom_model_dir)
% Step 4: Integrated Evaluation Script
% 本脚本用于在 MATLAB 环境下评估 Learned Dynamics 的多步预测效果。
%
% 用法:
%   verify_deployment                (自动加载最新的验证数据集和最新的模型)
%   verify_deployment(data_path)      (指定数据集，模型用最新的)
%   verify_deployment([], model_path)  (数据集用最新的，指定模型目录名/路径)
%   verify_deployment(data_path, model_path) (同时手动指定)

%% 1. 环境准备
PROJECT_ROOT = 'd:/Coding/Project/CPS/BlackBox-Simplex-Extension';
% 修改：根据当前日期时间创建唯一的报告文件夹，统一输出到 verification 目录下
report_timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
VERIFICATION_OUT = fullfile(PROJECT_ROOT, 'experiment/verification/out', report_timestamp);
if ~exist(VERIFICATION_OUT, 'dir'), mkdir(VERIFICATION_OUT); end
addpath(VERIFICATION_OUT); 
addpath(genpath('.')); 
TRAIN_OUT = fullfile(PROJECT_ROOT, 'experiment/train/out');

% 自动寻找最新的时间戳模型文件夹 (如果没有手动指定模型目录)
if nargin < 2 || isempty(custom_model_dir)
    d = dir(fullfile(TRAIN_OUT, '20*'));
    if isempty(d)
        error('在 %s 目录下未找到任何模型文件夹。', TRAIN_OUT);
    end
    [~, idx] = max([d.datenum]);
    LATEST_MODEL_DIR = fullfile(TRAIN_OUT, d(idx).name);
else
    if isfolder(custom_model_dir)
        LATEST_MODEL_DIR = custom_model_dir;
    else
        % 尝试在 TRAIN_OUT 寻找
        LATEST_MODEL_DIR = fullfile(TRAIN_OUT, custom_model_dir);
        if ~exist(LATEST_MODEL_DIR, 'dir')
            error('指定的模型目录不存在: %s', custom_model_dir);
        end
    end
end

%% 2. 加载模型及统计量
ONNX_PATH = fullfile(LATEST_MODEL_DIR, 'residual_model.onnx');
STATS_PATH = fullfile(LATEST_MODEL_DIR, 'scaling_stats.json');

if ~exist(ONNX_PATH, 'file'), error('ONNX 模型不存在: %s', ONNX_PATH); end
if ~exist(STATS_PATH, 'file'), error('统计量文件不存在: %s', STATS_PATH); end

fprintf('--> Using Model: %s\n', LATEST_MODEL_DIR);
fprintf('--> Loading ONNX: %s\n', ONNX_PATH);

% 核心：importONNXFunction 会返回 params 结构体，必须捕获并传递给生成的函数
params_onnx = importONNXFunction(ONNX_PATH, fullfile(VERIFICATION_OUT, 'residual_net'));

% 读取归一化参数
fid = fopen(STATS_PATH);
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
stats_raw = jsondecode(str);

% 核心修复：确保所有统计量都是 1xN 的行向量（Row Vectors）
stats.X_mean = reshape(double(stats_raw.X_mean), 1, []); 
stats.X_std  = reshape(double(stats_raw.X_std), 1, []);
stats.U_mean = reshape(double(stats_raw.U_mean), 1, []);
stats.U_std  = reshape(double(stats_raw.U_std), 1, []);
stats.R_mean = reshape(double(stats_raw.R_mean), 1, []);
stats.R_std  = reshape(double(stats_raw.R_std), 1, []);

%% 3. 数据测试路径逻辑
% 修改：支持命令行指定文件名
COLLECT_ROOT = fullfile(PROJECT_ROOT, 'traj', 'step3_collect');
if nargin > 0 && ~isempty(custom_val_mat)
    if isfolder(custom_val_mat)
        % 如果输入是目录，自动查找其下的 dataset_val.mat
        VAL_MAT = fullfile(custom_val_mat, 'dataset_val.mat');
        if ~exist(VAL_MAT, 'file')
            error('在目录 %s 下未找到 dataset_val.mat。', custom_val_mat);
        end
    elseif exist(custom_val_mat, 'file')
        VAL_MAT = custom_val_mat;
    else
        % 尝试在 collect_root 寻找
        VAL_MAT = fullfile(COLLECT_ROOT, custom_val_mat);
        if ~exist(VAL_MAT, 'file')
            error('指定的数据集文件或目录不存在: %s', custom_val_mat);
        end
    end
else
    % 默认逻辑：从最新的采集数据中取一条验证 case
    d2 = dir(fullfile(COLLECT_ROOT, '20*'));
    if isempty(d2)
        error('在 %s 目录下未找到任何采集数据。', COLLECT_ROOT);
    end
    % 修改：按文件夹名称(时间戳)排序，确保选择最新的数据集
    [~, sorted_idx] = sort({d2.name});
    idx2 = sorted_idx(end); 
    VAL_MAT = fullfile(COLLECT_ROOT, d2(idx2).name, 'dataset_val.mat');
end

fprintf('--> Using Dataset: %s\n', VAL_MAT);
fprintf('--> Evaluating Rollout Consistency...\n');

if exist(VAL_MAT, 'file')
    % 使用结构体接收 load 结果，避免变量被覆盖
    val_data = load(VAL_MAT); 
    X_val = val_data.split_ds.X;
    U_val = val_data.split_ds.U;
    R_label = val_data.split_ds.R_label;
    
    % --- 0. 准备测试参数 (适应新的 dynamics_learned 接口) ---
    % 核心修复：这个 params 必须与数据采集时的 true_dynamics 参数完全一致
    % 我们从第一个样本所属的轨迹文件中提取原始 params 结构体
    num_samples = size(X_val, 2);
    horizon = 10; % 定义多步预测步长
    results_table = struct('index', {}, 'error_norm', {}, 'avg_dist_err', {}, 'step_errors', {}, 'tag', {}, 'case_id', {});
    
    parent_dir = fileparts(VAL_MAT);
    manifest_path = fullfile(parent_dir, 'manifest.mat');
    origin_found = false;
    
    if exist(manifest_path, 'file')
        m_data = load(manifest_path);
        if isfield(m_data, 'cases')
            ref_case = m_data.cases(1);
            % 获取参数覆盖 (params_overrides) 用于报告打印备选
            ref_params = ref_case.params_overrides;
            
            try
                [m_folder, ~, ~] = fileparts(manifest_path);
                sample_traj_file = fullfile(m_folder, sprintf('case_%03d', ref_case.case_id), sprintf('traj_case_%03d.mat', ref_case.case_id));
                
                % 关键日志：打印正在尝试读取的文件路径（用于调试）
                fprintf('--> Debug: Extracting params from %s\n', sample_traj_file);
                
                if exist(sample_traj_file, 'file')
                    t_data = load(sample_traj_file);
                    if isfield(t_data, 'traj') && isfield(t_data.traj, 'params')
                        % 直接获取采集时的完整 params 结构体 (包含 dt, vmax, n, dmin, acc_scale 等)
                        eval_params = t_data.traj.params; 
                        origin_found = true;
                        fprintf('--> Debug: Successfully loaded params. acc_scale = %.3f\n', ...
                            isfield(eval_params,'acc_scale')*eval_params.acc_scale + ~isfield(eval_params,'acc_scale')*1.0);
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

    % 根据真实的 n 适配状态维度适配
    n_agents = eval_params.n;
    state_dim = 4 * n_agents;
    action_dim = 2 * n_agents;

    fprintf('--> Starting Full Validation Scan (%d samples)...\n', num_samples);
    
    % 为了速度，我们跳着采样（例如每隔 10 个 transition 采样一个），或者全扫
    scan_step = 1; 
    
    for i = 1:scan_step:num_samples - horizon
        % 确保 X_val 维度适配 (state_dim x N)
        current_x = double(X_val(:, i));
        if length(current_x) ~= state_dim
            current_x = double(X_val(i, :))';
        end
        
        x_start = reshape(current_x, 1, state_dim);
        
        % 控制序列同理处理 (action_dim x horizon)
        if size(U_val, 1) == action_dim
            u_seq = double(U_val(:, i:i+horizon-1))'; 
        else
            u_seq = double(U_val(i:i+horizon-1, :));
        end
        
        % 执行 10-step Rollout
        x_curr = x_start;
        current_step_errors = zeros(1, horizon);
        for t = 1:horizon
            [x_next_raw, ~] = dynamics_learned(x_curr, u_seq(t, :), @residual_net, params_onnx, stats, eval_params);
            x_curr = x_next_raw(1, :);
            
            % 计算当前步 (1到10) 的累积误差
            x_true_t = double(X_val(i + t, :)); 
            if size(X_val, 2) ~= state_dim, x_true_t = double(X_val(:, i + t))'; end
            current_step_errors(t) = norm(x_curr - x_true_t);
        end
        
        % 比较真实值（第 i+horizon 个样本是第 i 个样本经过 horizon 步后的结果）
        if size(X_val, 2) == 60
            x_true_final = double(X_val(i + horizon, :)); 
        else
            x_true_final = double(X_val(:, i + horizon))';
        end
        
        diff = x_curr - x_true_final;
        
        % 计算指标
        err_norm = norm(diff);
        
        % 计算机器人平均距离误差 (使用动态 n_agents)
        dx = diff(1:n_agents);
        dy = diff(n_agents+1:2*n_agents);
        avg_dist = mean(sqrt(dx.^2 + dy.^2));

        % 记录
        res.index = i;
        res.error_norm = err_norm;
        res.avg_dist_err = avg_dist;
        res.step_errors = current_step_errors; % 记录每一步的误差分布
        res.tag = 'unknown'; 
        res.case_id = 'N/A';
        
        results_table(end+1) = res;
        
        if mod(i, 100) == 0, fprintf('Processed %d/%d...\n', i, num_samples); end
    end

    %% 5. 生成分析报告
    [max_err, max_idx] = max([results_table.error_norm]);
    worst_sample = results_table(max_idx);
    
    report_file = fullfile(VERIFICATION_OUT, 'error_analysis_report.txt');
    fid = fopen(report_file, 'w');
    fprintf(fid, '====================================================\n');
    fprintf(fid, '         RESIDUAL MODEL ERROR ANALYSIS REPORT       \n');
    fprintf(fid, '         Generated on: %s\n', datestr(now));
    fprintf(fid, '         Model:        %s\n', LATEST_MODEL_DIR);
    fprintf(fid, '         Dataset:      %s\n', VAL_MAT);
    fprintf(fid, '====================================================\n\n');
    
    fprintf(fid, '1. OVERALL STATISTICS:\n');
    fprintf(fid, '   - Total Samples Validated: %d\n', length(results_table));
    fprintf(fid, '   - Mean L2 Norm Error:      %.6f\n', mean([results_table.error_norm]));
    fprintf(fid, '   - Median L2 Norm Error:    %.6f\n', median([results_table.error_norm]));
    fprintf(fid, '   - Max L2 Norm Error:       %.6f (Worst Case)\n\n', max_err);
    
    fprintf(fid, '2. WORST CASE DETAIL:\n');
    fprintf(fid, '   - Sample Index in Val Set: %d\n', worst_sample.index);
    fprintf(fid, '   - L2 Norm Error:           %.6f\n', worst_sample.error_norm);
    fprintf(fid, '   - Avg Agent Dist Error:    %.6f m\n', worst_sample.avg_dist_err);
    
    % --- 关键修复：打印物理参数误差标签 ---
    if origin_found
        fprintf(fid, '   - Physical Discrepancy (Ground Truth Settings From Data):\n');
        try
            % 核心修改：直接从 eval_params (即加载的 traj.params) 中读取字段
            if isfield(eval_params, 'acc_scale')
                fprintf(fid, '       * acc_scale: %.3f (Nominal: 1.000)\n', eval_params.acc_scale);
            end
            if isfield(eval_params, 'damping')
                fprintf(fid, '       * damping:   %.3f\n', eval_params.damping);
            end
            if isfield(eval_params, 'vmax')
                fprintf(fid, '       * vmax:      %.3f\n', eval_params.vmax);
            end
            if isfield(eval_params, 'dt')
                fprintf(fid, '       * dt:        %.3f\n', eval_params.dt);
            end
            if isfield(eval_params, 'n')
                fprintf(fid, '       * n_agents:  %d\n', eval_params.n);
            end
        catch
            fprintf(fid, '       * (Error displaying specific fields from eval_params)\n');
        end
    else
        fprintf(fid, '   - Physical Discrepancy: Manifest/Traj not found in %s\n', parent_dir);
    end
    fprintf(fid, '   - Suggestion: Check if this sample belongs to "boundary_focus" tag.\n\n');
    
    fprintf(fid, '3. ERROR DISTRIBUTION (L2 Norm):\n');
    % 简单的直方图统计
    edges = [0, 0.5, 1, 2, 5, 10, 50];
    counts = histcounts([results_table.error_norm], edges);
    for k = 1:length(counts)
        fprintf(fid, '   - [%.1f - %.1f]: %d samples\n', edges(k), edges(k+1), counts(k));
    end
    
    fclose(fid);
    fprintf('\n--> REPORT GENERATED: %s\n', report_file);
    fprintf('Worst Case Sample Index: %d, Error: %.6f\n', worst_sample.index, max_err);

    %% 6. 绘图：综合因果分析图表
    figure('Name', 'Causality Analysis', 'Visible', 'off'); 
    set(gcf, 'Position', [100, 100, 1200, 1000]); % 增加高度以容纳四个子图
    
    % --- 子图 1: 10步累积误差 (L2 Norm) ---
    subplot(4, 1, 1);
    plot([results_table.index], [results_table.error_norm], '-o', 'LineWidth', 1.5, 'Color', [0.85, 0.33, 0.1]);
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
    plot_file = fullfile(VERIFICATION_OUT, 'causality_analysis_plot.png');
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
    prop_plot_file = fullfile(VERIFICATION_OUT, 'error_propagation_plot.png');
    saveas(gcf, prop_plot_file);
    fprintf('--> ERROR PROPAGATION PLOT SAVED: %s\n', prop_plot_file);
    close(gcf);

else
    warning('Validation dataset not found.');
end

fprintf('\nStep 4 Part A & B: Integration & Rollout Finished.\n');
