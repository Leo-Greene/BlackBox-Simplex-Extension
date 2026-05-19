clc;
clear;
close all;

% relabel_old_dataset
% Recompute X_next_nominal and R_label with alpha mismatch for legacy datasets.

% ---- User config ----
output_root = fullfile('traj', 'step3_collect');
manual_dataset_dir = ''; % set to a specific dataset folder to override auto-detect
overwrite_files = true; % true to overwrite existing dataset_*.mat
% set numeric to override JSON/params
alpha_v_override = 0.85; 
alpha_x_override = 0.9; 

dt_override = [];      % set numeric to override JSON/params
% ---------------------

if isempty(manual_dataset_dir)
    files = dir(fullfile(output_root, '**', 'dataset_all.mat'));
    if isempty(files)
        error('No dataset_all.mat found under %s.', output_root);
    end
    [~, idx] = max([files.datenum]);
    dataset_dir = files(idx).folder;
else
    dataset_dir = manual_dataset_dir;
end

fprintf('--> Relabeling dataset in: %s\n', dataset_dir);

params = load_physical_params(dataset_dir);
if ~isempty(alpha_v_override), params.alpha_v = alpha_v_override; end
if ~isempty(alpha_x_override), params.alpha_x = alpha_x_override; end
if ~isempty(dt_override), params.dt = dt_override; end

% --- Debug Info ---
fprintf('\n--- Debug: Parameter Load Check ---\n');
if isfield(params, 'alpha_v')
    fprintf('  [OK] alpha_v: %.3f\n', params.alpha_v);
else
    fprintf('  [WARN] alpha_v: NOT FOUND in JSON (will default to 1.0 in relabeling)\n');
end

if isfield(params, 'alpha_x')
    fprintf('  [OK] alpha_x: %.3f\n', params.alpha_x);
else
    fprintf('  [WARN] alpha_x: NOT FOUND in JSON (will default to 1.0 in relabeling)\n');
end
fprintf('-----------------------------------\n\n');

relabel_one(fullfile(dataset_dir, 'dataset_all.mat'), params, overwrite_files, 'dataset');
relabel_one(fullfile(dataset_dir, 'dataset_train.mat'), params, overwrite_files, 'split_ds');
relabel_one(fullfile(dataset_dir, 'dataset_val.mat'), params, overwrite_files, 'split_ds');
relabel_one(fullfile(dataset_dir, 'dataset_test.mat'), params, overwrite_files, 'split_ds');

write_physical_params_json(params, dataset_dir);

fprintf('Relabel finished.\n');

function params = load_physical_params(dataset_dir)
params = struct();
json_path = fullfile(dataset_dir, 'physical_params.json');
if exist(json_path, 'file')
    raw = fileread(json_path);
    params = jsondecode(raw);
end
end

function relabel_one(mat_path, params, overwrite_files, var_name)
if ~exist(mat_path, 'file')
    return;
end
S = load(mat_path);
if ~isfield(S, var_name)
    warning('Missing %s in %s. Skip.', var_name, mat_path);
    return;
end

ds = S.(var_name);
[ds, params_out] = relabel_dataset_struct(ds, params);

if overwrite_files
    out_path = mat_path;
else
    [p, n, e] = fileparts(mat_path);
    out_path = fullfile(p, [n '_relabel' e]);
end

S.(var_name) = ds;
if strcmp(var_name, 'dataset')
    dataset = S.(var_name); %#ok<NASGU>
    save(out_path, 'dataset', '-v7.3');
else
    split_ds = S.(var_name); %#ok<NASGU>
    save(out_path, 'split_ds', '-v7.3');
end

fprintf('Relabeled: %s\n', out_path);
params = params_out; %#ok<NASGU>
end

function [ds, params_out] = relabel_dataset_struct(ds, params)
X = ds.X;
U = ds.U;
X_next_true = ds.X_next_true;

n = size(X, 2) / 4;
if ~isfield(params, 'n') || isempty(params.n)
    params.n = n;
end
if ~isfield(params, 'dt') || isempty(params.dt)
    params.dt = 0.3;
end
if ~isfield(params, 'alpha_v') || isempty(params.alpha_v)
    params.alpha_v = 1.0;
end
if ~isfield(params, 'alpha_x') || isempty(params.alpha_x)
    params.alpha_x = 1.0;
end

[pos_x, pos_y, vel_x, vel_y] = split_state(X, n);
[acc_x, acc_y] = split_action(U, n);

vel_next_x = params.alpha_v .* (vel_x + acc_x * params.dt);
vel_next_y = params.alpha_v .* (vel_y + acc_y * params.dt);

pos_next_x = pos_x + params.alpha_x .* (vel_next_x * params.dt);
pos_next_y = pos_y + params.alpha_x .* (vel_next_y * params.dt);

X_next_nominal = [pos_next_x, pos_next_y, vel_next_x, vel_next_y];
R_label = X_next_true - X_next_nominal;

ds.X_next_nominal = X_next_nominal;
ds.R_label = R_label;
if ~isfield(ds, 'X_true')
    ds.X_true = build_x_true(X_next_true, ds);
end

if ~isfield(ds, 'episode_id')
    if isfield(ds, 'case_id')
        ds.episode_id = ds.case_id;
    else
        ds.episode_id = ones(size(X, 1), 1);
    end
end
if ~isfield(ds, 'is_BC_active')
    ds.is_BC_active = false(size(X, 1), 1);
end

params_out = params;
end

function X_true = build_x_true(X_next_true, ds)
if isfield(ds, 'episode_id')
    ep = ds.episode_id;
elseif isfield(ds, 'case_id')
    ep = ds.case_id;
else
    ep = ones(size(X_next_true, 1), 1);
end

X_true = X_next_true;
for i = 2:size(X_next_true, 1)
    if ep(i) == ep(i - 1)
        X_true(i, :) = X_next_true(i - 1, :);
    end
end
end

function [pos_x, pos_y, vel_x, vel_y] = split_state(X, n)
pos_x = X(:, 1:n);
pos_y = X(:, n+1:2*n);
vel_x = X(:, 2*n+1:3*n);
vel_y = X(:, 3*n+1:4*n);
end

function [acc_x, acc_y] = split_action(U, n)
acc_x = U(:, 1:n);
acc_y = U(:, n+1:2*n);
end

function write_physical_params_json(p_struct, output_root)
json_path = fullfile(output_root, 'physical_params.json');
json_text = jsonencode(p_struct);
fid = fopen(json_path, 'w');
if fid < 0
    warning('Could not write physical_params.json to %s', output_root);
    return;
end
fwrite(fid, json_text, 'char');
fclose(fid);
end
