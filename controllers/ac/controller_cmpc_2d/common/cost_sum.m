function [ ret ] = cost_sum( u, pos, vel, params )
%% cmpc
    N = params.n;
    acc = u2acc(u, params.n, params.h_ac);
    control_steps = uint8(params.ct / params.dt);
    res = 0;
    for h = 1:params.h_ac
        a = acc(:,:,h)';
        for k = 1:control_steps
            [pos, vel] = dynamics(pos, vel, a, params);
        end
        res = res + fitness( pos, params );
    end
    ret = res/params.h_ac + sum(u.^2)/params.h_ac;
end

