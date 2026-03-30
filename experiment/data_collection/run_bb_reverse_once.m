function [traj, run_info] = run_bb_reverse_once(cfg)
% run_bb_reverse_once Run one reverse BB simulation with optional overrides.
%
% cfg fields (all optional):
%   - case_id: numeric or string id.
%   - seed: rng seed.
%   - enable_plot: true/false (default false).
%   - save_mat: true/false (default true).
%   - output_root: root folder for outputs (default 'traj/step3_collect').
%   - params_overrides: struct of parameter overrides.
%   - split: train/val/test label.

if nargin < 1
    cfg = struct();
end

if ~isfield(cfg, 'case_id')
    cfg.case_id = 1;
end
if ~isfield(cfg, 'seed')
    cfg.seed = [];
end
if ~isfield(cfg, 'enable_plot')
    cfg.enable_plot = false;
end
if ~isfield(cfg, 'save_mat')
    cfg.save_mat = true;
end
if ~isfield(cfg, 'output_root')
    cfg.output_root = fullfile('traj', 'step3_collect');
end
if ~isfield(cfg, 'params_overrides')
    cfg.params_overrides = struct();
end
if ~isfield(cfg, 'split')
    cfg.split = 'train';
end

if ~isempty(cfg.seed)
    rng(cfg.seed, 'twister');
end

%% Dependencies
addpath(genpath('controllers/ac/controller_cmpc_2d'));
addpath(genpath('controllers/bc/safety_controller'));
addpath(genpath('decision_module'));
addpath(genpath('extended_BBS'));
addpath('common');
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'dynamics'));

%% params (defaults from main_bb_reverse.m)
params.n = 15;
params.dt = 0.3;
params.ct = 0.3;
params.h_ac = 10;
params.h_bc = 10;
params.steps = 60;

params.amax = 1.5;
params.vmax = 2;

params.dmin = 1.7;

params.diameter = 10;
params.switch_step = 1;

params.ws = 10000;
params.wt = 10;

params.ws_bb = 3000;
params.w_orient = 20;

% Unmodeled params in true_dynamics (defaults, can be overridden by cfg.params_overrides)
% params.acc_scale = 0.95;
% params.acc_bias = 0;
% params.damping = 0.03;
% params.nonlinear_drift = false;
% params.noise_std = 0;
% params.control_noise_std = 0; % Optional small action perturbation for exploration cases.

% params.acc_scale = 0.85;
% params.acc_bias = 0;
% params.damping = 0.08;
% params.nonlinear_drift = false;
% params.noise_std = 0.005;
% params.control_noise_std = 0; % Optional small action perturbation for exploration cases.

params.acc_scale = 0.7;
params.acc_bias = 0.1;
params.damping = 0.15;
params.nonlinear_drift = true;
params.noise_std = 0.05;
params.control_noise_std = 0; % Optional small action perturbation for exploration cases.

params.predator = 0;
params.pFactor = 1.40;
params.pred_radius = 6;
params.wp = 500;

params = apply_overrides(params, cfg.params_overrides);

%% Optimizer settings
opt = optimoptions('fmincon');
opt.Display = 'off';
opt.MaxIterations = 8000;
opt.MaxFunctionEvaluations = 12000;
opt.FunctionTolerance = 1e-7;

%% Initial configuration
[posi, veli, params.tgt] = gen_init_bb(params);

%% Result buffers
x = zeros([params.steps, params.n]);
y = zeros([params.steps, params.n]);
vx = zeros([params.steps, params.n]);
vy = zeros([params.steps, params.n]);
ax = zeros([params.steps + 1, params.n]);
ay = zeros([params.steps + 1, params.n]);
mpc_cost = zeros(1, params.steps);
f = zeros(1, params.steps);
bb_sp = zeros(1, params.steps);
bb_orient = zeros(1, params.steps);
policy = ones(1, params.steps);

exit_flag_optimizer = zeros(1, params.steps);
a_sequence = zeros(params.steps, params.n, 2, params.h_bc + 1);
a_ac_traj = zeros(2, params.n, params.steps);

x(1, :) = posi(1, :);
y(1, :) = posi(2, :);
vx(1, :) = veli(1, :);
vy(1, :) = veli(2, :);
f(1) = fitness(posi, params);

bc_counter = 1;
rslt = [];
prev_seq = [];

%% Controller and dynamics
pos = posi;
vel = veli;
a_h = 0;
controller_run = params.ct / params.dt;
mde = 1;

tStart = tic;
for t = 1:params.steps
    if mod(t - 1, controller_run) == 0
        [a_ac, fval, e_flag, ~, ~] = controller_cmpc_2d(pos, vel, params, opt);

        [next_pos, next_vel] = next_state(pos, vel, a_ac, params);
        [~, ~, ~, ~, ~, a_h] = controller_safety_bb(next_pos, next_vel, params, opt);

        [decision, result] = decison_module(pos, vel, params, a_ac, a_h);

        if mde == 1
            if decision
                mde = 2;
                if isempty(prev_seq)
                    % No previous BC sequence available yet; hold position until a valid sequence is available.
                    prev_seq = a_h;
                    acc = zeros(size(a_ac));
                else
                    action_number = min(bc_counter, size(prev_seq, 3));
                    [acc, prev_seq] = resolve_collision(result, pos, vel, params, prev_seq, a_ac, a_h, action_number, t);
                end
                bc_counter = bc_counter + 1;

                params.switch_step = t;
                result.is_switch = decision;
                result.switch_step = t;
                rslt = [rslt result];
            else
                prev_seq = a_h;
                acc = a_ac;
            end
        else
            if decision
                if isempty(prev_seq)
                    prev_seq = a_h;
                    acc = zeros(size(a_ac));
                else
                    action_number = min(bc_counter, size(prev_seq, 3));
                    [acc, prev_seq] = resolve_collision(result, pos, vel, params, prev_seq, a_ac, a_h, action_number, t);
                end
                bc_counter = bc_counter + 1;
            else
                mde = 1;
                bc_counter = 1;
                prev_seq = a_h;
                acc = a_ac;
            end
        end
    else
        acc = [ax(t, :); ay(t, :)];
    end

    if isfield(params, 'control_noise_std') && params.control_noise_std > 0
        acc = acc + params.control_noise_std .* randn(size(acc));
    end

    [pos, vel] = true_dynamics(pos, vel, acc, params);

    x(t + 1, :) = pos(1, :);
    y(t + 1, :) = pos(2, :);
    vx(t + 1, :) = vel(1, :);
    vy(t + 1, :) = vel(2, :);
    ax(t + 1, :) = acc(1, :);
    ay(t + 1, :) = acc(2, :);

    a_ac_traj(:, :, t + 1) = a_ac;
    policy(t) = mde;
    a_sequence(t + 1, :, :, :) = a_h;
    mpc_cost(t) = fval;
    exit_flag_optimizer(t) = e_flag;
    f(t + 1) = fitness(pos, params);
    [~, bb_sp(t + 1), bb_orient(t + 1)] = fitness_bb(pos, vel, params);
end
runtime_s = toc(tStart);

%% Pack trajectory
traj.x = x;
traj.y = y;
traj.vx = vx;
traj.vy = vy;
traj.ax = ax(1:params.steps, :);
traj.ay = ay(1:params.steps, :);
% Applied actions aligned with transitions x_k -> x_{k+1}.
traj.ax_applied = ax(2:params.steps + 1, :);
traj.ay_applied = ay(2:params.steps + 1, :);
traj.a_ac = a_ac_traj;
traj.mpc_cost = mpc_cost;
traj.fitness = f;
traj.a_sequence = a_sequence;
traj.exit_flag = exit_flag_optimizer;
traj.params = params;
traj.bb_sp = bb_sp;
traj.bb_orient = bb_orient;
traj.result = rslt;
traj.policy = policy;

run_info = struct();
run_info.case_id = cfg.case_id;
run_info.seed = cfg.seed;
run_info.split = cfg.split;
run_info.runtime_s = runtime_s;
run_info.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
run_info.params_overrides = cfg.params_overrides;

traj.meta = run_info;

    %% Optional plotting
    if cfg.enable_plot
        displayTraj(x, y, vx, vy, policy);
        title(['Black-Box Simplex Case ', num2str(cfg.case_id)], 'FontSize', 17);
    end

    %% Save
    if cfg.save_mat
    % output_root already contains the unique timestamp from the calling script
    out_dir = fullfile(cfg.output_root, ['case_', pad_case_id(cfg.case_id)]);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    out_name = ['traj_case_' pad_case_id(cfg.case_id) '.mat'];
    save(fullfile(out_dir, out_name), 'traj');

    run_info.output_dir = out_dir;
    run_info.output_file = fullfile(out_dir, out_name);
end
end

function params = apply_overrides(params, overrides)
if isempty(overrides)
    return;
end

fns = fieldnames(overrides);
for i = 1:numel(fns)
    params.(fns{i}) = overrides.(fns{i});
end
end

function s = pad_case_id(case_id)
if isnumeric(case_id)
    s = sprintf('%03d', case_id);
else
    s = char(case_id);
end
end
