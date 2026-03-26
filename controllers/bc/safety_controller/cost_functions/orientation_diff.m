function [res] = orientation_diff(vel, params)
res = 0;
for i = 1:params.n
    for j = i+1:params.n
        v1 = vel(:,i);
        v2 = vel(:,j);
%         theta = acos(dot(v1,v2)/(norm(v1) * norm(v2)))^2;
%         res = res + theta;
        res = res + ( dot(v1,v2)/(norm(v1) * norm(v2)) + 1)^2;
        
    end
end
end

