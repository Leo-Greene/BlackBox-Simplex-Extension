function [a, fit_val, exit_flag, prev_sol, history] = controller_cmpc_2d(pos, vel, params, opt)
%% controller_cmpc_2d - Run MPC controller and return first action
% Non-liear constraints passed to fmincon as function handle. 
% like this: 
%     x = fmincon(@myfun,x0,A,b,Aeq,beq,lb,ub,@mycon)
% Input:
%   - pos          % 2 x n - state of the flock 
%   - vel          % 2 x n - state of the flock 
%   - params       % parameters

% Output:
%   - a            % 2 x n - The sequence of control action over the horizon 
%   - fit_val      % cost for the optimal sequence of accelerations.
% Usama Mehmood - Jan 2020

%% OutFunction

% Set up shared variables with out_function
history.x = [];
history.fval = [];

opt.OutputFcn = @out_function;

%% Optimization
    lb = -params.amax * ones(2*params.n*params.h_ac,1);
    ub = params.amax * ones(2*params.n*params.h_ac,1);

%     u_init = [init_point(2*params.n+1:end); (params.amax/sqrt(2)) * rand(2*params.n,1)];
%     u_init = zeros(2 * params.n * params.h_ac, 1);
    u_init = 2 * rand(2 * params.n * params.h_ac, 1) - 1;
    
    
%     rng default % For reproducibility
%     gs = GlobalSearch;
%     cost = @(u)cost_sum(u, pos, vel, params);
%     constraint = @(u)constraints(u, params);
%     problem = createOptimProblem('fmincon','x0',u_init,...
%     'objective',cost,'lb',lb,'ub',ub, 'nonlcon', constraint, 'options', opt);
%     [u, fit_val] = run(gs,problem);
%     exit_flag = 0;
    
    [u, fit_val, exit_flag] = fmincon(@(u)cost_sum(u, pos, vel, params),u_init,[],[],[],[],lb,ub,@(u)constraints(u, params),opt);
%     [u, fit_val, exit_flag] = multi_start(pos, vel, params, opt, 4);

%% Solution is processed
    a_h = u2acc(u, params.n, params.h_ac);
    a = a_h(:,:,1)';
    for i = 1:params.n
        a(:,i) = trim_vec(a(:,i), params.amax);
    end
    prev_sol = u;
    
%% Function to collect the optimizers history. Used in plotting the fval.
    function stop = out_function(x,optimValues,state)
        stop = false;
        switch state
            case 'iter'
                history.fval = [history.fval; optimValues.fval];
                history.x = [history.x; x];
            otherwise
        end
    end
end

