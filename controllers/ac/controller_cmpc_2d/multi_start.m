function [u, fit_val, exit_flag] = multi_start(pos, vel, params, opt, num_points)
lb = -params.amax * ones(2*params.n*params.h,1);
ub = params.amax * ones(2*params.n*params.h,1);

inits = 2 * rand(num_points, 2 * params.n * params.h) - 1;
inits(1,:) = zeros(1, 2 * params.n * params.h);

for i = 1:num_points
    [sol(i).u, sol(i).fit_val, sol(i).exit_flag] = fmincon(@(u)cost_sum(u, pos, vel, params),inits(i,:)',[],[],[],[],lb,ub,@(u)constraints(u, params),opt);
end

%% Choose Lowest cost 
[fit_val, I] =  min([sol(:).fit_val]);
u = sol(I).u;
exit_flag = sol(I).exit_flag;

%% Display result
% disp(num2str(I));
a = [sol(:).fit_val];
b = [sol(:).u];
b = b(randi(params.n*params.h*2, 1, 10), :);
disp(b);

end

