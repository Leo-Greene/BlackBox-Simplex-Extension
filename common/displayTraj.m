function [] = displayTraj( x, y ,vx, vy, switches, collision_info)
    % --- Initial states (Green) ---
    displayInitState( x, y, vx, vy, 1, [0, 0.8, 0])
    
    % --- Axis Limits & Setup ---
    xlim([-21 21])
    ylim([-21 35])
    axis equal
    hold on;
    
    % --- Policy-Based Coloring ---
    num_agents = size(x, 2);
    num_steps = length(switches);
    color_p1 = [0.1, 0.5, 0.8]; % Nominal (Blueish)
    color_p2 = [1.0, 0.4, 0.2]; % Safety (Orange/Red)
    
    for j = 1:num_agents
        for t = 1:num_steps
            p = switches(t);
            c = color_p1;
            if p == 2, c = color_p2; end
            plot([x(t,j), x(t+1,j)], [y(t,j), y(t+1,j)], 'LineWidth', 1.0, 'Color', c);
        end
    end
    
    % --- Collision Highlighting ---
    h_col = [];
    if nargin > 5 && ~isempty(collision_info)
        % collision_info has fields: step, agent_i, agent_j, pos_i, pos_j
        for k = 1:numel(collision_info)
            c = collision_info(k);
            % Mark agents involved with red X
            h_col = plot(c.pos_i(1), c.pos_i(2), 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
            plot(c.pos_j(1), c.pos_j(2), 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
            % Draw connecting line for more clarity
            line([c.pos_i(1) c.pos_j(1)], [c.pos_i(2) c.pos_j(2)], 'Color', 'r', 'LineStyle', ':', 'LineWidth', 0.8);
        end
    end
    
    % --- Final states (Red) ---
    displayInitState( x, y, vx, vy, size(x, 1), [1,0,0])
    
    % --- Legend Construction ---
    h1 = plot(NaN, NaN, 'LineWidth', 2, 'Color', color_p1);
    h2 = plot(NaN, NaN, 'LineWidth', 2, 'Color', color_p2);
    
    leg_handles = [h1, h2];
    leg_labels = {'Policy 1 (AC)', 'Policy 2 (BC)'};
    
    if ~isempty(h_col)
        leg_handles = [leg_handles, h_col];
        leg_labels = [leg_labels, {'Collision Point'}];
    end
    
    legend(leg_handles, leg_labels, 'Location', 'northwest', 'FontSize', 10);

    set(gcf, 'Position', [400 400 600 600])
end

