
function [decision, result] = decison_module(pos, vel, params, a_ac, a_seq)

decision = false;
result = [];

%% One step sanity check for advanced controller command
[is_collision, next_pos, next_vel, r, c, d] = check_next_state(pos, vel, a_ac, params);
if is_collision
    disp(['[DM] [AC collision] Immediate collision detected between agents ' num2str(r(1)) ' and ' num2str(c(1)) ' with distance ' num2str(d)]);
    decision = is_collision;
    result.cause = 1;
    result.pair = [r(1), c(1), d];
    return;
end

%% m step sanity check for basleine controller command sequence
seq_len = size(a_seq, 3);
for i = 1:seq_len
    [is_collision, next_pos, next_vel, r, c, d] = check_next_state(next_pos, next_vel, a_seq(:,:,i)', params);
    if is_collision
        disp(['[DM] [BC collision] Collision detected at step ' num2str(i) ' between agents ' num2str(r(1)) ' and ' num2str(c(1)) ' with distance ' num2str(d)]);
        decision = is_collision;
        result.cause = 2;
        result.pair = [r(1), c(1), d, i];
        return;
    end
end

%% divergence check for the final state
[is_converging, r, c] = check_divergence_simple(next_pos, next_vel);
if is_converging
    disp(['[DM] Soft prompt: final velocities are not diverging between agents ' num2str(r(1)) ' and ' num2str(c(1))]);
    % Soft prompt only: keep the divergence warning for diagnostics,
    % but do not hard-reject the AC command sequence here.
    % decision = is_converging;
    % result.pair = [r(1), c(1)];
    % result.cause = 3;
end

end


