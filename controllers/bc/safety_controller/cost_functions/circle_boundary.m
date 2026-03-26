function [res] = circle_boundary(pos, params)
%% terminal_cost(pos, params)
    
    res = sum((params.diameter - sqrt(sum(pos.^2, 1))).^2 );

end

