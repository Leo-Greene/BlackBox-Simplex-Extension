function [res] = fitness(pos, params)
%#codegen
%% cmpc - fitness 

%% separation
sp = separation_polynomial(pos);

%% cohesion
% asd = sum_sq_distances( pos );

%% target cost
circle_tgt = sum_sq_distances_tgt(pos, params.tgt);


% res = asd + 1000000 * spC;
res = params.wt * circle_tgt...
    + params.ws * sp;

end

