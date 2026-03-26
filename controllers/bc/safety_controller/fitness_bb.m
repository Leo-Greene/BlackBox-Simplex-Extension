function [res, sp, orient_s] = fitness_bb(pos, vel, params)
%#codegen
%% cmpc - fitness 

sp = separation_polynomial(pos);
% safety_circle = circle_boundary(pos, params);
% orient_d = orientation_diff(vel, params);
orient_s = orientation_single(pos, vel, params);

res = params.ws_bb * sp + params.w_orient * orient_s;

end

