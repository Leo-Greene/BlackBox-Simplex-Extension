function [c, ceq] = constraints_bb(u, params)
% constraints - non-linear constraints to enforce magnitude of acceleration
%%
acc = u2acc(u, params.n, params.h_ac);
res = sqrt(sum(acc.^2,2)) - params.amax;
c = res(:);
ceq = [];
end

