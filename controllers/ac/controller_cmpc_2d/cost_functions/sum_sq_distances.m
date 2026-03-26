function [ res ] = sum_sq_distances( pos )
%% cmpc Given an pos -> 2 x n matrix, res is sum of squared pairwise distances. 
%%
    res = 0;
    n = size(pos, 2);
    for i = 1:n-1
        for j = i+1:n
            res = res + sum((pos(:,i) - pos(:,j)).^2);
        end
    end
    count = n * (n - 1) / 2;
    res = res / count;
end

