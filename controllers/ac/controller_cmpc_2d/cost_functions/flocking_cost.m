function [ret] = flocking_cost(pos, params)

spC = separation_constrained(pos, params);
asd = sum_sq_distances( pos );

ret = asd + params.rho * spC;
end

