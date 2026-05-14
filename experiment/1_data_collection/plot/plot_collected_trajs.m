clc;
clear;
close all;

% plot_collected_trajs.m
% Enhanced plotting script for Step 3 Collected Trajectories.
% Supports parallel execution and collision marking.

%% 1. Path Setup
SCRIPT_DIR = fileparts(mfilename('fullpath'));
PROJECT_ROOT = fullfile(SCRIPT_DIR, '..', '..', '..');
addpath(fullfile(PROJECT_ROOT, 'common'));
addpath(genpath(fullfile(PROJECT_ROOT, 'common', 'display')));
addpath(fullfile(PROJECT_ROOT, 'extended_BBS')); % for displayTraj
addpath(fullfile(PROJECT_ROOT, 'decision_module'));
addpath(fullfile(PROJECT_ROOT, 'experiment', 'other')); % for check_collision

output_root = fullfile(PROJECT_ROOT, 'traj', 'step3_collect');
manifest_mat = '';

% --- 自动寻找最新的 manifest.mat (基于文件夹名称中的时间戳) ---
if isempty(manifest_mat)
    % 获取所有符合日期格式 (20xx-xx-xx_xxxxxx_default) 的文件夹
    % 注意：数据采集可能会有多种后缀，所以这里放宽匹配
    d = dir(output_root);
    d = d([d.isdir]);
    names = {d.name};
    % 匹配 yyyy-mm-dd_HHMMSS 格式
    valid_idx = ~cellfun(@isempty, regexp(names, '^\d{4}-\d{2}-\d{2}_\d{6}.*$'));
    if ~any(valid_idx)
        error('未找到符合时间戳命名的 Step 3 结果文件夹于: %s', output_root);
    end
    
    % yyyy-mm-dd_HHMMSS 格式天然支持字符串排序，排序后最后一个即为最新
    valid_names = sort(names(valid_idx));
    latest_folder = valid_names{end};
    manifest_mat = fullfile(output_root, latest_folder, 'manifest.mat');
    
    if ~exist(manifest_mat, 'file')
        error('最新的文件夹 [%s] 中未找到 manifest.mat。', latest_folder);
    end
end

[source_folder, ~, ~] = fileparts(manifest_mat);
fprintf('正在为采集结果作图: %s\n', source_folder);

%% 2. Load Manifest
S = load(manifest_mat);
manifest = S.manifest;
num_cases = numel(manifest);

%% 3. Parallel Setup
force_parallel = true;
use_parallel = force_parallel && license('test', 'Distrib_Computing_Toolbox');
if use_parallel
    pool = gcp('nocreate');
    if isempty(pool), parpool; end
end

%% 4. Loop and Plot
if use_parallel
    fprintf('--> 并行绘图启动 (针对集成结果风格)... \n');
    parfor i = 1:num_cases
        if ~strcmp(manifest(i).status, 'ok')
            continue;
        end
        
        mat_file = manifest(i).output_file;
        % Handle cases where output_file might be relative in old manifests
        if ~exist(mat_file, 'file')
            [~, file_name, file_ext] = fileparts(mat_file);
            mat_file = fullfile(source_folder, ['case_', sprintf('%03d', manifest(i).case_id)], [file_name, file_ext]);
        end
        if ~exist(mat_file, 'file'), continue; end
        
        % Load Data
        data = load(mat_file);
        traj = data.traj;
        
        % Render in hidden figure
        h = figure('Visible', 'off');
        [collision_info, ~] = check_collision(traj);
        displayTraj(traj.x, traj.y, traj.vx, traj.vy, traj.policy, collision_info);
        
        case_id_str = sprintf('%03d', manifest(i).case_id);
        
        % Title & Style
        is_collision = false;
        % Some step 3 manifests might not have has_collision pre-calculated accurately
        if isfield(manifest, 'has_collision')
            is_collision = manifest(i).has_collision; 
        else
            is_collision = ~isempty(collision_info);
        end
        
        tag_str = '';
        if isfield(manifest(i), 'tag'), tag_str = [' (', manifest(i).tag, ')']; end
        
        if is_collision
            num_c = numel(collision_info);
            title(['[COLLISION] Collected Case ', case_id_str, tag_str, ' (Collisions: ', num2str(num_c), ')'], 'Color', 'r', 'FontSize', 14);
            img_name_suffix = '_COLLISION.png';
        else
            title(['[SAFE] Collected Case ', case_id_str, tag_str], 'Color', [0 0.5 0], 'FontSize', 14);
            img_name_suffix = '.png';
        end
        
        [case_dir, ~, ~] = fileparts(mat_file);
        img_name = fullfile(case_dir, ['plot_case_', case_id_str, img_name_suffix]);
        
        set(h, 'PaperPositionMode', 'auto');
        print(h, img_name, '-dpng', '-r200');
        close(h);
        
        fprintf('Done: Case %s\n', case_id_str);
    end
else
    for i = 1:num_cases
        if ~strcmp(manifest(i).status, 'ok'), continue; end
        mat_file = manifest(i).output_file;
        if ~exist(mat_file, 'file'), continue; end
        
        fprintf('正在处理 Case %d/%d (Case ID: %d)...\n', i, num_cases, manifest(i).case_id);
        data = load(mat_file);
        traj = data.traj;
        
        h = figure('Visible', 'off');
        [collision_info, ~] = check_collision(traj);
        displayTraj(traj.x, traj.y, traj.vx, traj.vy, traj.policy, collision_info);
        
        case_id_str = sprintf('%03d', manifest(i).case_id);
        is_collision = ~isempty(collision_info);
        
        tag_str = '';
        if isfield(manifest(i), 'tag'), tag_str = [' (', manifest(i).tag, ')']; end
        
        if is_collision
            num_c = numel(collision_info);
            title(['[COLLISION] Collected Case ', case_id_str, tag_str, ' (Collisions: ', num2str(num_c), ')'], 'Color', 'r', 'FontSize', 14);
            img_name_suffix = '_COLLISION.png';
        else
            title(['[SAFE] Collected Case ', case_id_str, tag_str], 'Color', [0 0.5 0], 'FontSize', 14);
            img_name_suffix = '.png';
        end
        
        [case_dir, ~, ~] = fileparts(mat_file);
        img_name = fullfile(case_dir, ['plot_case_', case_id_str, img_name_suffix]);
        set(h, 'PaperPositionMode', 'auto');
        print(h, img_name, '-dpng', '-r200');
        close(h);
    end
end

fprintf('\n数据采集绘图任务完成。图片已保存至对应的轨迹文件夹中。\n');
fprintf('目录: %s\n', source_folder);
