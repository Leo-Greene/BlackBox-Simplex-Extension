function [report_text, summary_data] = analyze_results(results, num_runs)
%ANALYZE_RESULTS Summarize batch run results for None vs PrSBC.

if nargin < 2
    num_runs = size(results, 1);
end

none_stats = init_stats();
prsbc_stats = init_stats();

for i = 1:num_runs
    none_stats = accumulate_stats(none_stats, results{i, 1});
    prsbc_stats = accumulate_stats(prsbc_stats, results{i, 2});
end

summary_data.none = finalize_stats(none_stats, num_runs);
summary_data.prsbc = finalize_stats(prsbc_stats, num_runs);

report_text = build_report(summary_data, num_runs);
end

function stats = init_stats()
stats.run_count = 0;
stats.collision_runs = 0;
stats.collision_steps = 0;
stats.min_dist = [];
stats.mean_min_dist = [];
stats.mean_cost = [];
stats.ctrl_dev = [];
stats.goal_success = 0;
stats.goal_tol = [];
stats.safe_radius = [];
end

function stats = accumulate_stats(stats, traj)
if isempty(traj)
    return;
end

stats.run_count = stats.run_count + 1;

[min_dist_trace, min_dist_overall] = compute_min_dist(traj);
collision_mask = min_dist_trace < get_dmin(traj);

stats.collision_steps = stats.collision_steps + sum(collision_mask);
if any(collision_mask)
    stats.collision_runs = stats.collision_runs + 1;
end

stats.min_dist(end+1) = min_dist_overall;
stats.mean_min_dist(end+1) = mean(min_dist_trace);

if isfield(traj, 'mpc_cost')
    stats.mean_cost(end+1) = mean(traj.mpc_cost);
end

stats.ctrl_dev(end+1) = compute_control_deviation(traj);

[goal_success, goal_tol] = compute_goal_success(traj);
if ~isnan(goal_success)
    stats.goal_success = stats.goal_success + goal_success;
    stats.goal_tol = goal_tol;
end

stats.safe_radius = get_safe_radius(traj);
end

function stats = finalize_stats(stats, num_runs)
if stats.run_count == 0
    return;
end

stats.collision_rate = stats.collision_runs / stats.run_count;
stats.collision_steps_mean = stats.collision_steps / stats.run_count;
stats.min_dist_mean = mean(stats.min_dist);
stats.min_dist_min = min(stats.min_dist);
stats.mean_min_dist_mean = mean(stats.mean_min_dist);

if ~isempty(stats.mean_cost)
    stats.mean_cost_mean = mean(stats.mean_cost);
else
    stats.mean_cost_mean = NaN;
end

if ~isempty(stats.ctrl_dev)
    stats.ctrl_dev_mean = mean(stats.ctrl_dev);
else
    stats.ctrl_dev_mean = NaN;
end

if stats.goal_success > 0 && ~isempty(stats.goal_tol)
    stats.goal_success_rate = stats.goal_success / stats.run_count;
else
    stats.goal_success_rate = NaN;
end

stats.run_count = num_runs;
end

function report_text = build_report(summary, num_runs)
header = sprintf('PrSBC Batch Test Report\n');
header = [header sprintf('Runs: %d\n', num_runs)];
header = [header sprintf('Timestamp: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'))];

row_fmt = '%-26s | %-14s | %-14s | %-10s\n';

lines = sprintf(row_fmt, 'Metric', 'None', 'PrSBC', 'Delta');
lines = [lines sprintf('%s\n', repmat('-', 1, 72))];

lines = [lines metric_line(row_fmt, 'Collision rate', summary.none.collision_rate, summary.prsbc.collision_rate, true)];
lines = [lines metric_line(row_fmt, 'Collision steps/run', summary.none.collision_steps_mean, summary.prsbc.collision_steps_mean, false)];
lines = [lines metric_line(row_fmt, 'Min distance (min)', summary.none.min_dist_min, summary.prsbc.min_dist_min, false)];
lines = [lines metric_line(row_fmt, 'Min distance (mean)', summary.none.mean_min_dist_mean, summary.prsbc.mean_min_dist_mean, false)];
lines = [lines metric_line(row_fmt, 'Control deviation', summary.none.ctrl_dev_mean, summary.prsbc.ctrl_dev_mean, false)];
lines = [lines metric_line(row_fmt, 'Mean MPC cost', summary.none.mean_cost_mean, summary.prsbc.mean_cost_mean, false)];

if ~isnan(summary.none.goal_success_rate) || ~isnan(summary.prsbc.goal_success_rate)
    lines = [lines metric_line(row_fmt, 'Goal success rate', summary.none.goal_success_rate, summary.prsbc.goal_success_rate, true)];
end

report_text = [header lines];
end

function line = metric_line(fmt, label, none_val, prsbc_val, is_percent)
if is_percent
    none_str = format_percent(none_val);
    prsbc_str = format_percent(prsbc_val);
    delta_val = prsbc_val - none_val;
    delta_str = format_percent(delta_val);
else
    none_str = format_number(none_val);
    prsbc_str = format_number(prsbc_val);
    delta_val = prsbc_val - none_val;
    delta_str = format_number(delta_val);
end

line = sprintf(fmt, label, none_str, prsbc_str, delta_str);
end

function [min_dist_trace, min_dist_overall] = compute_min_dist(traj)
if ~isfield(traj, 'x') || ~isfield(traj, 'y')
    min_dist_trace = NaN;
    min_dist_overall = NaN;
    return;
end

x = traj.x;
y = traj.y;
steps = size(x, 1);
min_dist_trace = zeros(steps, 1);

for t = 1:steps
    pos = [x(t, :); y(t, :)];
    min_dist_trace(t) = min_pairwise_distance(pos);
end

min_dist_overall = min(min_dist_trace);
end

function dmin = min_pairwise_distance(pos)
% pos is 2 x n
n = size(pos, 2);
if n < 2
    dmin = NaN;
    return;
end

min_val = inf;
for i = 1:n-1
    for j = i+1:n
        d = norm(pos(:, i) - pos(:, j));
        if d < min_val
            min_val = d;
        end
    end
end

dmin = min_val;
end

function dmin = get_dmin(traj)
if isfield(traj, 'params') && isfield(traj.params, 'dmin')
    dmin = traj.params.dmin;
else
    dmin = 1.0;
end
end

function r_safe = get_safe_radius(traj)
if isfield(traj, 'params') && isfield(traj.params, 'R_safe')
    r_safe = traj.params.R_safe;
else
    r_safe = NaN;
end
end

function dev = compute_control_deviation(traj)
if ~isfield(traj, 'a_ac') || ~isfield(traj, 'ax') || ~isfield(traj, 'ay')
    dev = NaN;
    return;
end

acc = cat(1, permute(traj.ax, [3 2 1]), permute(traj.ay, [3 2 1]));
a_ac = traj.a_ac;
steps = min(size(a_ac, 3), size(acc, 3));

if steps == 0
    dev = NaN;
    return;
end

dev_sum = 0;
count = 0;
for t = 1:steps
    diff = acc(:, :, t) - a_ac(:, :, t);
    dev_sum = dev_sum + mean(vecnorm(diff, 2, 1));
    count = count + 1;
end

dev = dev_sum / max(count, 1);
end

function [success, goal_tol] = compute_goal_success(traj)
if ~isfield(traj, 'params') || ~isfield(traj.params, 'tgt')
    success = NaN;
    goal_tol = NaN;
    return;
end

if isfield(traj.params, 'dmin')
    goal_tol = 0.5 * traj.params.dmin;
else
    goal_tol = 0.5;
end

pos_final = [traj.x(end, :); traj.y(end, :)];
tgt = traj.params.tgt;

if isempty(tgt) || size(tgt, 2) ~= size(pos_final, 2)
    success = NaN;
    return;
end

final_dist = vecnorm(pos_final - tgt, 2, 1);
success = all(final_dist <= goal_tol);
end

function s = format_number(val)
if isnan(val)
    s = 'n/a';
else
    s = sprintf('%.4f', val);
end
end

function s = format_percent(val)
if isnan(val)
    s = 'n/a';
else
    s = sprintf('%.2f%%', 100 * val);
end
end
