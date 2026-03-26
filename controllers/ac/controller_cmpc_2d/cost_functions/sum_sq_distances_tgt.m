function [res] = sum_sq_distances_tgt(pos, tgt)
%SUM_SQ_DISTANCES_TGT Summary of this function goes here

res = sum(sum((pos - tgt).^2));
end

