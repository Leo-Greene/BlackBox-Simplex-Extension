function [res] = target(pos, target, params)
res = 0;
for i = 1:params.n
    res = res + norm(pos(:,i) - target);
end
res = res / params.n;
end

