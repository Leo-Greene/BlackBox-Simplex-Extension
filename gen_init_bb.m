function [posi, veli, tgt] = gen_init_bb(params)
%% gen_init_bb - Generate initial positions for the multiagent systems.
% Initial positions are on a circle with equal separation between the
% agents.

%%
n = params.n;
theta_offset = 0;
if isfield(params, 'init_theta_offset_deg')
	theta_offset = params.init_theta_offset_deg;
end

theta = 0:360/n:360*(n-1)/n;
theta = theta + theta_offset;

posi = [params.diameter * cosd(theta); params.diameter * sind(theta)];

if isfield(params, 'init_pos_jitter_std') && params.init_pos_jitter_std > 0
	posi = posi + params.init_pos_jitter_std .* randn(size(posi));
end

target_rotation_deg = 180;
if isfield(params, 'target_rotation_deg')
	target_rotation_deg = params.target_rotation_deg;
end

R = [cosd(target_rotation_deg) -sind(target_rotation_deg); sind(target_rotation_deg) cosd(target_rotation_deg)];
tgt = R * posi;

veli = zeros(2, params.n);
if isfield(params, 'init_vel_noise_std') && params.init_vel_noise_std > 0
	veli = veli + params.init_vel_noise_std .* randn(size(veli));
end

end

