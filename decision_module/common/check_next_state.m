function [is_collision, next_pos, next_vel, r, c, dist] = check_next_state(cur_pos, cur_vel, cur_acc, params)
r = 0;
c = 0;
dist = 0;

is_collision = false;
control_steps = uint8(params.ct / params.dt);

if isfield(params, 'use_learned_dynamics') && params.use_learned_dynamics
    n = params.n;
    % Flatten data formats to match Neural Network expected 1D input shapes
    x_vec = [cur_pos(1,:), cur_pos(2,:), cur_vel(1,:), cur_vel(2,:)];
    u_vec = [cur_acc(1,:), cur_acc(2,:)];
    
    for k = 1:control_steps
        [x_next_vec, ~] = dynamics_learned(x_vec, u_vec, ...
            params.learned_model.func, params.learned_model.params_onnx, params.learned_model.stats, params);
        x_vec = x_next_vec;
    end
    
    % Unflatten outputs back to default standard dimensions for existing distance equations
    cur_pos(1,:) = x_vec(1:n);
    cur_pos(2,:) = x_vec(n+1:2*n);
    cur_vel(1,:) = x_vec(2*n+1:3*n);
    cur_vel(2,:) = x_vec(3*n+1:4*n);
else
    for k = 1:control_steps
        [cur_pos, cur_vel] = dynamics(cur_pos, cur_vel, cur_acc, params);
    end
end

next_pos = cur_pos;
next_vel = cur_vel;

d = inter_agent_distance(next_pos);
condition = d < params.dmin & eye(params.n) == 0;
if any(any(condition))
    is_collision = true;
    [r,c] = find(condition);
    r = r(1);
    c = c(1);
    dist = d(r,c);
end
end

