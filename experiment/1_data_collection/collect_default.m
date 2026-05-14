clc;
clear;
close all;

% collect_default.m
% 该脚本专门用于生成全默认配置的数据集（无物理参数扰动）。
% 核心保持与 collect.m 一致的 48-case 结构，但所有 case 均使用默认配置。

num_cases = 48;
base_seed = 260318;
stamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
output_root = fullfile('traj', 'step3_collect', [stamp, '_default']);
force_parallel = true;
requested_workers = 0; 

use_parallel = force_parallel && license('test', 'Distrib_Computing_Toolbox');

% 保持随机种子生成方式一致，但不再生成扰动
rng(base_seed, 'twister');

cases = repmat(struct(), num_cases, 1);

for i = 1:num_cases
    cases(i).case_id = i;
    cases(i).seed = base_seed + i;
    cases(i).save_mat = true;
    cases(i).enable_plot = false;
    cases(i).output_root = output_root;
    cases(i).params_overrides = struct(); % 强制为空结构体，即完全使用默认值
    cases(i).tag = 'default';
end

% --- 实现分层抽样 (虽然只有 default 一个 tag，但保持切分比例一致) ---

train_ratio = 0.70;
val_ratio = 0.15;

n_train = round(num_cases * train_ratio);
n_val = round(num_cases * val_ratio);

for k = 1:num_cases
    if k <= n_train
        cases(k).split = 'train';
    elseif k <= (n_train + n_val)
        cases(k).split = 'val';
    else
        cases(k).split = 'test';
    end
end

if ~exist(output_root, 'dir')
    mkdir(output_root);
end

manifest = repmat(struct( ...
    'case_id', 0, ...
    'seed', 0, ...
    'split', '', ...
    'tag', '', ...
    'runtime_s', NaN, ...
    'status', '', ...
    'message', '', ...
    'output_file', ''), num_cases, 1);

fprintf('Default Collection started | parallel=%d | num_cases=%d\n', use_parallel, num_cases);

if use_parallel
    pool = gcp('nocreate');
    if isempty(pool)
        if requested_workers > 0
            parpool(requested_workers);
        else
            parpool;
        end
    end

    manifest_local = repmat(manifest(1), num_cases, 1);
    parfor i = 1:num_cases
        manifest_local(i) = execute_case(cases(i));
    end
    manifest = manifest_local;
else
    for i = 1:num_cases
        fprintf('Running case %d/%d | split=%s\n', i, num_cases, cases(i).split);
        manifest(i) = execute_case(cases(i));
    end
end

manifest_mat = fullfile(output_root, 'manifest.mat');
save(manifest_mat, 'manifest', 'cases');

manifest_csv = fullfile(output_root, 'manifest.csv');
T = struct2table(manifest);
writetable(T, manifest_csv);

fprintf('\nDefault Collection finished.\n');
fprintf('Output path: %s\n', output_root);

function row = execute_case(case_cfg)
row = struct( ...
    'case_id', case_cfg.case_id, ...
    'seed', case_cfg.seed, ...
    'split', case_cfg.split, ...
    'tag', case_cfg.tag, ...
    'runtime_s', NaN, ...
    'status', '', ...
    'message', '', ...
    'output_file', '');

try
    [~, info] = run_bb_reverse_once(case_cfg);
    row.runtime_s = info.runtime_s;
    row.status = 'ok';
    row.message = '';
    if isfield(info, 'output_file')
        row.output_file = info.output_file;
    end
catch ME
    row.status = 'failed';
    row.message = ME.message;
end
end
