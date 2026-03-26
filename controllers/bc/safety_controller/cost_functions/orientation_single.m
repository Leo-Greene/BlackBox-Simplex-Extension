function [res] = orientation_single(pos, vel, params)
res = 0;
    for i = 1:params.n
        v1 = vel(:,i);
        v2 = pos(:,i);
        res = res + (1 - dot(v1,v2)/(norm(v1) * norm(v2)))^2;
    end
end

