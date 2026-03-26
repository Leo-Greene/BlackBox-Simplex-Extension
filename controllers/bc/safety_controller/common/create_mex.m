%% code gen cost_sum( u, pos, vel, params )

params_type = coder.typeof(params);
u_type = coder.typeof(zeros(2*params.n*params.h_bc,1));
pos_type = coder.typeof(zeros(2,params.n));
vel_type = coder.typeof(zeros(2,params.n));

codegen cost_sum_bb.m -args {u_type, pos_type, vel_type, params_type} -o cost_sum_bb

%% code gen angle_vectors(u,v)

% vec_type = coder.typeof([0;0;0]);
% 
% codegen angle_vectors.m -args {vec_type, vec_type} -o angle_vectors

%% code gen constraints(u, params)

u_type = coder.typeof(zeros(2*params.n*params.h_bc,1));
params_type = coder.typeof(params);

codegen constraints_bb.m -args {u_type, params_type} -o constraints_bb



