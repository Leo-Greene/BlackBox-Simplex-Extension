function [res] = predator_avoidance(pos, params)
%#codegen
%% cmpc - predator_avoidance

%% cohesion and separation of the flock
sp = separation_polynomial(pos(:,1:end-1));
asd = sum_sq_distances(pos(:,1:end-1));

%% predator cost

sq_dist_to_pred = sqrt(sum((repmat(pos(:,end), 1, params.n-1) - pos(:,1:params.n -1)).^2, 1)).^3;
pa = 0;
for i = 1:params.n - 1
%     if sq_dist_to_pred(i) < params.pred_radius^2
        pa = pa + 1/sq_dist_to_pred(i);
%     end
end

% pa = sum(1./ sq_dist_to_pred);

res = params.wc * asd...
    + params.ws * sp...
    + params.wp * pa;

end

