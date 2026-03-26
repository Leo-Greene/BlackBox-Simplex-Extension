
function [acc_out, prev_seq_out, counter, switch_step] = resolve_collision(result, pos, vel, params, prev_seq, a_ac, a_h, action_number, t)

acc_out = a_ac;
prev_seq_out = a_h;
decision = true;
counter = 0;
visited = false(1, params.n);
switch_step = params.switch_step;

if isempty(prev_seq)
    % Defensive fallback: if there is no previous sequence, issue zero command.
    acc_out = zeros(size(a_ac));
    return;
end

if nargin < 8 || isempty(action_number) || action_number < 1
    action_number = 1;
end
action_number = min(action_number, size(prev_seq, 3));

while(decision)
    
    c1 = int64(result.pair(1));
    c2 = int64(result.pair(2));

    % avoid loop
    if  visited(c1(1)) == 1 &&  visited(c2(1)) == 1
        disp('!!!! Decision not resolved; Performing full reverse switch !!!!');
        % Full reverse switch
        prev_seq_out = prev_seq;
        acc_out = prev_seq(:, :, action_number)';
        break
    end

    if visited(c1(1)) == 0
        prev_seq_out(c1(1),:,:) = prev_seq(c1(1), :, :);
        acc_out(:,c1(1)) = prev_seq(c1(1), :, action_number)';
        visited(c1(1)) = 1;
        switch_step(c1(1), t) = 1;
        counter = counter + 1;

    elseif visited(c2(1)) == 0
        prev_seq_out(c2(1),:,:) = prev_seq(c2(1), :, :);
        acc_out(:,c2(1)) = prev_seq(c2(1), :, action_number)';
        visited(c2(1)) = 1;
        switch_step(c2(1), t) = 1;
        counter = counter + 1;
    end

    i = min(action_number+1, size(prev_seq_out, 3));
    [decision, result] = decison_module(pos, vel, params, acc_out, prev_seq_out(:, :, i:size(prev_seq_out, 3)));
end

