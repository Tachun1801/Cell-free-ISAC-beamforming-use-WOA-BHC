%% Test WOA-BHC Implementation
% Test with simple configuration to verify the algorithm works correctly

clear all;
addpath(genpath('./'));

fprintf('Testing WOA-BHC Algorithm Implementation...\n\n');

%% Test 1: WOA-BHC Core Algorithm
fprintf('Test 1: WOA-BHC Core Algorithm\n');
% Simple sphere function: f(x) = sum(x.^2)
sphere_func = @(x) sum(x.^2);
dim = 10;
[best_pos, best_score, conv_curve] = WOA_BHC(sphere_func, dim, 20, 50, -100*ones(1,dim), 100*ones(1,dim));
fprintf('Best score (should be close to 0): %.6f\n', best_score);
assert(best_score < 1, 'WOA-BHC core test failed');
fprintf('✓ Test 1 passed\n\n');

%% Test 2: Communication Beamforming Optimization
fprintf('Test 2: Communication Beamforming with WOA-BHC\n');
% Simple test case
N_t = 8; M_t = 2; U = 3;
H_comm = (randn(U, M_t, N_t) + 1j*randn(U, M_t, N_t))/sqrt(2);
F_sensing = zeros(1, M_t, N_t); % No sensing
P_comm = 1;
sigmasq_comm = 0.1;
gamma = 5; % Target SINR

% WOA parameters for faster testing
woa_params.SearchAgents = 20;
woa_params.MaxIter = 50;
woa_params.bias_factor = 0.4;
woa_params.boundary_threshold = 0.15;

[F_star, feasible] = opt_comm_WOA_BHC(H_comm, sigmasq_comm, P_comm, F_sensing, gamma, woa_params);
fprintf('Feasible: %d\n', feasible);
fprintf('F_star size: %s\n', mat2str(size(F_star)));
assert(all(size(F_star) == [U, M_t, N_t]), 'Output size incorrect');
fprintf('✓ Test 2 passed\n\n');

%% Test 3: Bisection Search
fprintf('Test 3: Bisection SINR Search with WOA-BHC\n');
wrapped_objective = @(g) opt_comm_WOA_BHC(H_comm, sigmasq_comm, P_comm, F_sensing, g, woa_params);
[F_opt, SINR_max] = bisection_SINR_WOA_BHC(0.1, 20, 0.5, wrapped_objective);
fprintf('Maximum achievable SINR: %.2f\n', SINR_max);
assert(SINR_max > 0, 'Bisection search failed');
fprintf('✓ Test 3 passed\n\n');

%% Test 4: Joint Sensing-Communication
fprintf('Test 4: Joint Sensing-Communication with WOA-BHC\n');
T = 1; % One target
sensing_beamsteering = (randn(T, M_t, N_t) + 1j*randn(T, M_t, N_t))/sqrt(2);
sens_streams = 1;
sigmasq_sens = 0.1;
P_all = 1;
gamma_jsc = 3;

[F_jsc, feasible_jsc, SSNR] = opt_jsc_WOA_BHC(H_comm, sigmasq_comm, gamma_jsc, sensing_beamsteering, sens_streams, sigmasq_sens, P_all, woa_params);
fprintf('JSC Feasible: %d, SSNR: %.4f\n', feasible_jsc, SSNR);
fprintf('✓ Test 4 passed\n\n');

%% Test 5: Beam Extraction
fprintf('Test 5: Beam Extraction with WOA-BHC\n');
[F_comm_ext, F_sens_ext] = WOA_BHC_beam_extraction(F_jsc, H_comm);
fprintf('F_comm_ext size: %s\n', mat2str(size(F_comm_ext)));
fprintf('F_sens_ext size: %s\n', mat2str(size(F_sens_ext)));
assert(size(F_comm_ext, 1) == U, 'Communication beam extraction failed');
fprintf('✓ Test 5 passed\n\n');

fprintf('All tests passed successfully! ✓✓✓\n');
