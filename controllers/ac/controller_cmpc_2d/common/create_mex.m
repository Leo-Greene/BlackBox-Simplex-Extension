%% code gen cost_sum( u, pos, vel, params )

params_type = coder.typeof(params);
u_type = coder.typeof(zeros(2*params.n*params.h_ac,1));
pos_type = coder.typeof(zeros(2,params.n));
vel_type = coder.typeof(zeros(2,params.n));

codegen cost_sum.m -args {u_type, pos_type, vel_type, params_type} -o cost_sum

%% code gen angle_vectors(u,v)

% vec_type = coder.typeof([0;0;0]);
% 
% codegen angle_vectors.m -args {vec_type, vec_type} -o angle_vectors

%% code gen constraintsconstraints(u, params)

u_type = coder.typeof(zeros(2*params.n*params.h_ac,1));
params_type = coder.typeof(params);

codegen constraints.m -args {u_type, params_type} -o constraints



