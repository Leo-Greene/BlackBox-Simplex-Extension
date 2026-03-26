clc
clear all

%% Parameters
params.n = 2;
params.dmin = 2.0;
params.rho = 1000000;

%%
pos = [0 2;...
       0 2];
% pos -> [x1 x2;...
%         y1 y2];

[ret] = flocking_cost(pos, params);