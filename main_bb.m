clc
clear all
close all

%% Dependencies
% root = '/Users/Usama/UmehmoodGoogle/Work/Code/';

addpath(genpath('safety_controller'));
addpath(genpath('controller_cmpc_2d'));
addpath(genpath('decision_module'));

addpath('common');


% addpath([root 'Swarms New/m-functions']);

%% params

params.n = 7;
params.dt = 0.3;
params.ct = 0.3;
params.h_ac = 10;
params.h_bc = 10;
params.steps = 40;

params.amax = 1.5; %1.5
params.vmax = 2;%2

% Safety property 
params.dmin = 1.7;

% inital configuration
params.diameter = 10;
params.switch_step = 1;

params.ws = 10000; %1800
params.wt = 10;

params.ws_bb = 3000; %1800
params.w_orient = 20;

%% Predator params
params.predator = 0;
params.pFactor = 1.40;
params.pred_radius = 6;
params.wp = 500;

%% Results structure
result.is_switch = false;

%% Optimizer Settings
opt = optimoptions('fmincon');
opt.Display = 'off';
% opt.Algorithm = 'active-set';
opt.MaxIterations = 8000;
opt.MaxFunctionEvaluations = 12000;
% opt.StepTolerance = 1e-12;
% Stopping criteria if change in function value is lesser than this:
opt.FunctionTolerance = 1e-7;


%%
% u = zeros(2*params.n*params.h + 1,1);
indexes = 1:params.n;
acc = zeros(2, params.n);
controller_run = params.ct / params.dt;
zero_vec = zeros(params.n , 1) ;
params.switch_step = params.steps;

cpu_time = 0;
function_calls = 0;
tic;

%% Initial configuration

[posi, veli, params.tgt] = gen_init_bb(params);

% scatter(posi(1,:), posi(2,:),'.', 'MarkerEdgeColor', 'r', 'SizeData', 200);
% hold on
% scatter(params.tgt(1,:), params.tgt(2,:), 'x', 'MarkerEdgeColor', 'b', 'SizeData', 100);
% axis equal
% 
% run controller_cmpc_2d/common/create_mex.m;
% run safety_controller/common/create_mex.m;

%%
x = zeros([params.steps, params.n]);
y = zeros([params.steps, params.n]);
vx = zeros([params.steps, params.n]);
vy = zeros([params.steps, params.n]);
ax = zeros([params.steps+1, params.n]);
ay = zeros([params.steps+1, params.n]);
bd = zeros([params.n, params.steps, params.n]);
mpc_cost = zeros(1, params.steps);
f = zeros(1, params.steps); 
bb_sp = zeros(1, params.steps); 
bb_orient = zeros(1, params.steps); 

exit_flag_optimizer = zeros(1, params.steps);
a_sequence = zeros(params.steps, params.n, 2, params.h_bc+1);

a_ac_traj = zeros(2, params.n, params.steps);

x(1,:) = posi(1,:);
y(1,:) = posi(2,:);
vx(1,:) = veli(1,:);
vy(1,:) = veli(2,:);
f(1) = fitness(posi, params);
decision = false;
bc_counter = 1;

%% Controller and Dynamics:
pos = posi; vel = veli;
prev_sol = zeros(2*params.n*params.h_ac,1);
a_h = 0;
for t = 1:params.steps
    if mod(t,5) == 0
        e = round(toc, 1);
        disp(['step: ' num2str(t) '/' num2str(params.steps) ', Time:' num2str(e) 's']);
    end
    
    if mod(t - 1, controller_run) == 0
        if ~decision
            [a_ac, fval, e_flag, prev_sol, history] = controller_cmpc_2d(pos, vel, params, opt);
            
            [next_pos, next_vel] = next_state(pos, vel, a_ac, params);
            [~, ~, ~, ~, ~, a_h] = controller_safety_bb(next_pos, next_vel, params, opt);
            
            [decision, result] = decison_module(pos, vel, params, a_ac, a_h);
            if decision
                params.switch_step = t;
                result.is_switch = decision;
                result.switch_step = t;
            end
            if ~decision
                prev_seq = a_h;
                acc = a_ac;
            end
        end

        if decision 
            action_number = min(bc_counter, size(prev_seq, 3));
            acc = prev_seq(:,:,action_number)';
            bc_counter = bc_counter + 1;
            
%             [acc, fval, e_flag, prev_sol, history, a_h] = controller_safety_bb(pos, vel, params, opt);
%         plot(history.fval, 'LineWidth', 1.5, 'Marker', '*', 'Color', 'r', 'MarkerEdgeColor', 'b');
%         pause(0.1);
        end
    else
        acc = [ax(t, :); ay(t, :)];
    end
    
    [pos, vel] = dynamics(pos, vel, acc, params);
    x(t+1,:) = pos(1,:);
    y(t+1,:) = pos(2,:);
    vx(t+1,:) = vel(1,:);
    vy(t+1,:) = vel(2,:);
    ax(t+1,:) = acc(1,:);
    ay(t+1,:) = acc(2,:);
    a_ac_traj(:, :, t+1) = a_ac;
%     if t >= params.switch_step
        a_sequence(t+1,:,:,:) = a_h;
%     end
    mpc_cost(t) = fval;
    exit_flag_optimizer(t) = e_flag;
    f(t+1) = fitness(pos, params);
    [~, bb_sp(t+1), bb_orient(t+1)] = fitness_bb(pos, vel, params);
end

%% Store output.
traj.x = x;
traj.y = y;
traj.vx = vx;
traj.vy = vy;
traj.ax = ax(1:params.steps,:);
traj.ay = ay(1:params.steps,:);
traj.a_ac = a_ac_traj;
traj.mpc_cost = mpc_cost;
traj.fitness = f;
traj.a_sequence = a_sequence;
traj.exit_flag = exit_flag_optimizer;
traj.params = params;
traj.bb_sp = bb_sp;
traj.bb_orient = bb_orient;
traj.result = result;

displayTraj(x,y,vx,vy, params.switch_step); title('Black-Box Simplex', 'FontSize', 17);
set(gcf, 'Position', get(0, 'Screensize'));
% saveas(gcf, ['Images/trajComparison_' num2str(i) '.jpg']);


%% Save output

date_string = datestr(datetime,' [yyyy-mm-dd]');
out_traj_name = ['traj_bb'  date_string];
 
dir_path = 'D:\Coding\Project\MAS\BlackBox-Simplex-Extension\traj\'; % 输出路径

[dest_path, out_traj_name] = create_dir(dir_path, out_traj_name);
mkdir([dest_path '\Results']);

fprintf('\nOUTPUT:\n%s\n', dest_path);
fprintf(['switch: ' num2str(params.switch_step) '\n']);
if numel(result)
    disp(result_message(result, params.dt));
else
    disp('No switching')
end


save([dest_path '/' out_traj_name '.mat'], 'traj');

%%
figure;
plot(traj.bb_orient)





