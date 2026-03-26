clc;
clear;
close all;

% merge_step3_datasets
% 本脚本用于合并多个采集文件夹（含有 manifest.mat）的数据。
% 
% 使用说明：
%   1. 在下面的 manifest_list 中手动填入需要合并的文件夹路径。
%   2. 脚本会创建一个带有 _MERGED 后缀的新文件夹（例如 2026-03-20_120000_MERGED）。
%   3. 该新文件夹包含合并后的 dataset_*.mat 和 manifest.mat。
%   4. Python 训练脚本会自动识别这个最新生成的文件夹进行训练。

output_root = fullfile('traj', 'step3_collect');
% 核心：将 common 目录加入路径以包含 dynamics.m
addpath(fullfile(pwd, 'common')); 

% --- 请在下面手动填入你想合并的文件夹名称 ---
manifest_list = { ...
    fullfile(output_root, '2026-03-18_172620', 'manifest.mat'), ...
    fullfile(output_root, '2026-03-18_182225', 'manifest.mat')  ...
};

% 创建一个新的合并输出文件夹
export_output_root = fullfile(output_root, [datestr(now, 'yyyy-mm-dd_HHMMSS'), '_MERGED']);
if ~exist(export_output_root, 'dir'), mkdir(export_output_root); end

X = [];
U = [];
X_next_true = [];
X_next_nominal = [];
R_label = [];
case_id = [];
step_id = [];
split = {};
tag = {};
all_manifests = []; 

fprintf('Starting dataset merge...\n');

for m = 1:numel(manifest_list)
    manifest_mat = manifest_list{m};
    if ~exist(manifest_mat, 'file')
        warning('Manifest missing, skipping: %s', manifest_mat);
        continue;
    end
    
    fprintf('--> Processing manifest: %s\n', manifest_mat);
    S = load(manifest_mat);
    manifest = S.manifest;
    all_manifests = [all_manifests; manifest]; 

    for i = 1:numel(manifest)
        if ~strcmp(manifest(i).status, 'ok'), continue; end
        
        data_file = manifest(i).output_file;
        if ~exist(data_file, 'file')
            [m_dir, ~, ~] = fileparts(manifest_mat);
            [~, d_name, d_ext] = fileparts(data_file);
            data_file = fullfile(m_dir, [d_name, d_ext]);
        end

        if ~exist(data_file, 'file')
            warning('Skip case %d: output file missing.', manifest(i).case_id);
            continue;
        end

        T = load(data_file);
        if ~isfield(T, 'traj'), continue; end
        traj = T.traj;

        [Xi, Ui, Xni_true, Xni_nom, Ri, cid, sid] = trajectory_to_transitions(traj, manifest(i).case_id);

        n_i = size(Xi, 1);
        X = [X; Xi];
        U = [U; Ui];
        X_next_true = [X_next_true; Xni_true];
        X_next_nominal = [X_next_nominal; Xni_nom];
        R_label = [R_label; Ri];
        case_id = [case_id; cid];
        step_id = [step_id; sid];
        split = [split; repmat({manifest(i).split}, n_i, 1)];
        tag = [tag; repmat({manifest(i).tag}, n_i, 1)];
    end
end

% 保存合并后的 manifest 以便 Python 识别
manifest = all_manifests; 
save(fullfile(export_output_root, 'manifest.mat'), 'manifest');

dataset = struct();
dataset.X = X;
dataset.U = U;
dataset.X_next_true = X_next_true;
dataset.X_next_nominal = X_next_nominal;
dataset.R_label = R_label;
dataset.case_id = case_id;
dataset.step_id = step_id;
dataset.split = split;
dataset.tag = tag;
dataset.feature_layout.state = '[x(1..n), y(1..n), vx(1..n), vy(1..n)]';
dataset.feature_layout.action = '[ax(1..n), ay(1..n)]';

all_file = fullfile(export_output_root, 'dataset_all.mat');
save(all_file, 'dataset', '-v7.3');

save_split(dataset, 'train', export_output_root);
save_split(dataset, 'val', export_output_root);
save_split(dataset, 'test', export_output_root);

fprintf('\nMerge finished.\n');
fprintf('Merged folder: %s\n', export_output_root);
fprintf('Samples total: %d\n', size(dataset.X, 1));

% --- 辅助函数 ---
function [X, U, X_next_true, X_next_nominal, R, cid, sid] = trajectory_to_transitions(traj, case_id_value)
    params = traj.params;
    x = traj.x; y = traj.y; vx = traj.vx; vy = traj.vy;
    if isfield(traj, 'ax_applied') && isfield(traj, 'ay_applied')
        ax = traj.ax_applied; ay = traj.ay_applied;
    else
        ax = traj.ax; ay = traj.ay;
    end
    T = min([size(x, 1) - 1, size(ax, 1)]);
    X = zeros(T, 4 * params.n);
    U = zeros(T, 2 * params.n);
    X_next_true = zeros(T, 4 * params.n);
    X_next_nominal = zeros(T, 4 * params.n);
    R = zeros(T, 4 * params.n);
    for k = 1:T
        % 获取当前时刻真实状态
        pos_k = [x(k, :); y(k, :)];
        vel_k = [vx(k, :); vy(k, :)];
        acc_k = [ax(k, :); ay(k, :)];
        
        % 获取下一时刻真实状态 (k+1)
        pos_k1_true = [x(k+1, :); y(k+1, :)];
        vel_k1_true = [vx(k+1, :); vy(k+1, :)];
        
        % 使用名义动力学模型重新计算名义预测值
        % 既然 traj 数据中没有预存，我们通过 dynamics.m 现场计算
        [pos_k1_nom, vel_k1_nom] = dynamics(pos_k, vel_k, acc_k, params);

        xk = [pos_k(1, :), pos_k(2, :), vel_k(1, :), vel_k(2, :)];
        uk = [acc_k(1, :), acc_k(2, :)];
        xk1_true = [pos_k1_true(1, :), pos_k1_true(2, :), vel_k1_true(1, :), vel_k1_true(2, :)];
        xk1_nom = [pos_k1_nom(1, :), pos_k1_nom(2, :), vel_k1_nom(1, :), vel_k1_nom(2, :)];

        X(k, :) = xk;
        U(k, :) = uk;
        X_next_true(k, :) = xk1_true;
        X_next_nominal(k, :) = xk1_nom;
        R(k, :) = xk1_true - xk1_nom;
    end
    cid = case_id_value * ones(T, 1); sid = (1:T)';
end

function save_split(dataset, split_name, output_root)
    mask = strcmp(dataset.split, split_name);

    split_ds = struct();
    split_ds.X = dataset.X(mask, :);
    split_ds.U = dataset.U(mask, :);
    split_ds.X_next_true = dataset.X_next_true(mask, :);
    split_ds.X_next_nominal = dataset.X_next_nominal(mask, :);
    split_ds.R_label = dataset.R_label(mask, :);
    split_ds.case_id = dataset.case_id(mask, :);
    split_ds.step_id = dataset.step_id(mask, :);
    split_ds.split = dataset.split(mask, :);
    split_ds.tag = dataset.tag(mask, :);
    split_ds.feature_layout = dataset.feature_layout;

    out_file = fullfile(output_root, ['dataset_' split_name '.mat']);
    save(out_file, 'split_ds', '-v7.3');

    fprintf('%s samples: %d -> %s\n', split_name, size(split_ds.X, 1), out_file);
end
