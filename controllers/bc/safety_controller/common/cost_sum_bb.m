function [ ret ] = cost_sum_bb( u, pos, vel, params )
%% cmpc
    N = params.n;
    acc = u2acc(u, params.n, params.h_bc);
    control_steps = uint8(params.ct / params.dt);
    res = 0;
    for h = 1:params.h_bc
        a = acc(:,:,h)';
        for k = 1:control_steps
            [pos, vel] = dynamics(pos, vel, a, params);
        end
        res = res + fitness_bb( pos, vel, params );
    end
    ret = res/params.h_bc + sum(u.^2)/params.h_bc;
end

