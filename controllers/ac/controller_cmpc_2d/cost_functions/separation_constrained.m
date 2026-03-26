function [res] = separation_constrained(pos, params)
    res = 0;
    for i = 1:params.n - 1
        for j = i+1:params.n
            res = res + max(params.dmin - norm(pos(:,i) - pos(:,j)), 0)^2;
        end
    end
    res = sqrt(res);
end

