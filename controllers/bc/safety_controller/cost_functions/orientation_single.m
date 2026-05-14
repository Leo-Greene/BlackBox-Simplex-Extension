function [res] = orientation_single(pos, vel, params)
res = 0;
    for i = 1:params.n
        v1 = vel(:,i);
        v2 = pos(:,i);
        v1_norm = norm(v1);
        v2_norm = norm(v2);
        if v1_norm < 1e-12    % 速度为0时，惩罚为1，位置为0时，惩罚为0
            res = res + 1;
        elseif v2_norm < 1e-12
            res = res;
        else
            res = res + (1 - dot(v1,v2)/(v1_norm * v2_norm))^2;
        end
    end
end

