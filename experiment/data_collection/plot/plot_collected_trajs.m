clc;
clear;
close all;

% plot_collected_trajs.m
% 必要的运行条件：
%   1. 确保当前工作目录位于项目根目录。
%   2. 确保 common/ 等绘图依赖路径在 MATLAB 路径中。
%   3. 脚本将自动寻找最新的 manifest.mat 或处理指定的目录。

%% 路径设置
addpath('common');
addpath(genpath('common/display'));
addpath('extended_BBS'); % 包含 display_reverse_switch.m
addpath('decision_module');

output_root = fullfile('traj', 'step3_collect');
manifest_mat = '';

% 自动寻找最新的 manifest.mat
if isempty(manifest_mat)
    files = dir(fullfile(output_root, '**', 'manifest.mat'));
    if isempty(files)
        error('未找到 manifest.mat 文件，请先运行数据采集脚本。');
    end
    [~, idx] = max([files.datenum]);
    manifest_mat = fullfile(files(idx).folder, files(idx).name);
end

[source_folder, ~, ~] = fileparts(manifest_mat);
fprintf('正在为目录下的轨迹作图: %s\n', source_folder);

%% 加载清单
S = load(manifest_mat);
manifest = S.manifest;

%% 遍历并绘图
num_cases = numel(manifest);
for i = 1:num_cases
    if ~strcmp(manifest(i).status, 'ok')
        fprintf('跳过 Case %d (状态: %s)\n', manifest(i).case_id, manifest(i).status);
        continue;
    end
    
    mat_file = manifest(i).output_file;
    if ~exist(mat_file, 'file')
        warning('文件不存在，跳过: %s', mat_file);
        continue;
    end
    
    % 加载数据
    fprintf('正在处理 Case %d/%d...\n', i, num_cases);
    data = load(mat_file);
    traj = data.traj;
    
    % 创建隐藏窗口绘图
    h = figure('Visible', 'off');
    displayTraj(traj.x, traj.y, traj.vx, traj.vy, traj.policy);
    
    case_str = sprintf('%03d', manifest(i).case_id);
    title(['Black-Box Simplex Case ', case_str, ' (', manifest(i).tag, ')'], 'FontSize', 14);
    
    % 保存图片
    [case_dir, ~, ~] = fileparts(mat_file);
    img_name = fullfile(case_dir, ['plot_case_', case_str, '.png']);
    saveas(h, img_name);
    close(h);
end

fprintf('\n绘图任务完成。图片已保存至对应的 case 文件夹中。\n');
