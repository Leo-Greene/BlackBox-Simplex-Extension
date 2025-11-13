tic
clc
clear;
close all;

%% add paths
addpath(genpath('common'));

%% Some on/off options
Description = ''; % Added to the output folder name.

selected_agents = [1,2];
pair_agents = [1,2]; % For lie-derivative and barrier function plot in simplex. 

skip_1 = 0; %How many steps to skip in video. 

Speed = 1; %0:Off, 1:on, 2:selective
speed_agent = selected_agents;
v_scale = 1.0;

Acceleration = 0; %0:Off, 1:on, 2:selective
Acc_scale = 1;
Acc_agent = selected_agents;


drawTraj = 1; %1: Incremental, 2: trail
trail_len = 1500;

showVideo = 1;
showText = 1;
FullBox = 1;
BoxL = 2;
showID = 1;
showCircles = 0;

PlotEdges = 0; %0:Off, 1:on, 2:selective
PlotEdgeId = selected_agents;
pbf = false;

simplex_policy = 1; % = 0 sanaz: because traj does not have policy field
traj_color_policy = 1;

simulation_starting_state = 1; %video starts from

BB_targets_plot = 1;
BB_plot_sequence = 0;
BB_show_vel = 1;
BB_show_acc = 0;

BB_draw_circle = 1;
BB_show_failed_seq = 1;
BB_fail_pause_duration = 0;

%% Data source
traj_num = 1; 
%traj_path = 'C:\Users\sanaz\Documents\Projects\BB-simplex-master\traj\traj_bb [2022-11-09]';
traj_path = 'traj\traj_bb [2023-01-24]_1'; % 15 agent simple COMMENT by lzj: 这个路径是运行main_bb_reverse之后数据保存的路径，每次都要手动修改



[newStr, matches] = split(traj_path, '\');
traj_name = [cell2mat(newStr(end)), '.mat']; 

save_to = [traj_path '\Results\'];
mkdir(save_to);
destPath = save_to;

%% Load trajectory Data - from matlab code, save files, passes on: x, y, vx, vy, accx, accy, policy
load([traj_path '\' traj_name]);
x = traj(traj_num).x;
y = traj(traj_num).y;
vx = traj(traj_num).vx;
vy = traj(traj_num).vy;
accx = traj(traj_num).ax;
accy = traj(traj_num).ay;
params = traj(traj_num).params;
%     params.rs = params.rs_bc;
mpc_cost = traj.mpc_cost;
exit_flag = traj.exit_flag;
a_sequence = traj.a_sequence;
a_ac = traj.a_ac;
% fitness = traj.fitness; %non-mpc fitness, computed at each time step.

if simplex_policy
    policy = traj(traj_num).policy;
     policy(1,:) = 1;
end

cur_date = datestr(datetime,'(yyyy-mm-dd, hh-MM)');
dirName = [cur_date ' ' num2str(params.n) '_' num2str(params.steps) '_' num2str(traj_num) '_' Description];
mkdir([save_to dirName]);

%% Set Video options 
videoName = [destPath dirName '/' dirName '-2'];
myVideo = VideoWriter(videoName,"Uncompressed AVI");
fastForwardRate = 0.7;
myVideo.FrameRate = min(30,fastForwardRate * (1/params.dt));
%myVideo.Quality = 50;
open(myVideo);
skip = skip_1 + 1;


%% draw and save trajectory
fig_traj = figure;
displayTraj( x, y ,vx, vy, params.switch_step);
saveas(gcf, [destPath dirName '/' 'trajectory' '.jpg']);

%% Graphs
f1 = figure;
rows = 5;
i = 1;

%% pairwise distance min & max
subplot(rows,2,i);
H = PlotDistance(x, y, params.steps, params );
axis([0 params.steps 0 5]);
i = i + 1;

%% Acc
subplot(rows,2,i);
plotAcc(accx, accy, params);
i = i + 1;

%% minimum distance - zoomed in veiw
subplot(rows,2,i);
min1 = plotMinPWdist(x, y, params); 
set(min1, 'Color', 'r');
plot([1, params.steps], [params.dmin params.dmin], 'g--', 'LineWidth', 1.5);
axis([0 params.steps params.dmin - 0.002 params.dmin + 0.002]);
i = i + 1;

%% Fitness - mpc
subplot(rows,2,i);
plotCurve(mpc_cost, 'Time', 'cost_{mpc}', 1.5, params.n);
i = i + 1;

%% Fitness - non mpc
% if data_source == 1
%     subplot(rows,2,i);
%     plotCurve(fitness, 'Time', 'Fitness', 1.5, params.n);
%     i = i + 1;
% end

%% Vel
subplot(rows,2,i);
plotVel( vx, vy );
i = i + 1;

%% exit flag for fmincon
subplot(rows,2,i);
plotCurve(exit_flag, 'Time', 'exit_flag', 1.5, params.n);
i = i + 1;

%% Policy
if simplex_policy
    subplot(rows, 2, i);
    plotCurve(policy(:,1)', 'Time', 'Policy', 1.5, params.n);
    ylim([0,3])
    i = i + 1;
end


%% Number of neighbors
% subplot(rows,2,i);
% for rs = [params.rs_bc]
%     num_of_neighbors = zeros(1, params.steps);
%     for index = 1:params.steps
%         q = [x(index,:)', y(index,:)'];
%         pnet = proximityNet(q, rs);
%         num_of_neighbors(index) = numberOfNeighbors(pnet);
%     end
%     plotCurve(num_of_neighbors, 'Time', ['mean num neighbors, rs = ' num2str(rs)], 1.5, params.n);
%     hold on
% end
% i = i + 1;

%% Save Plots
saveas(f1, [destPath dirName '/' 'combined_plots' '.jpg']);

savefig(f1, [destPath dirName '/' 'combined_plots']);

%% Show Video
if showVideo
    style = '--';
    traj_color = [0.8 0.8 0.8];
    make_video;
end
close(myVideo);
drawnow;





