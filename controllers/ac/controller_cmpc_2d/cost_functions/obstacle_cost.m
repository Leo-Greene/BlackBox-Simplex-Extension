function [res] = obstacle_cost(pos, rects, params)
%% obstacle_cost - The cost function for obstacle avoidance.

%%
res = 0;
for i = 1:params.n
    for j = 1:numel(rects)
        [d, ~] = point2rect(pos(:,i)', rects(j));
        if d ~= 0
            res = res + 1/d^2;
        end
    end
end

count = params.n * numel(rects);
res = res / count;

end

