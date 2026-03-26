function [res] = velocity_matching(vel, params) 
% Given an vel -> 2 x n matrix.
res = 0;
for i = 1:params.n-1
    for j = i+1:params.n
        res = res + norm(vel(:,i) - vel(:,j));
    end
end
count = params.n * (params.n - 1) / 2;
res = res / count;
end

