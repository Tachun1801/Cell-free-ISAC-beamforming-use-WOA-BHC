function [F_star, feasible] = opt_comm_WOA_BHC(H_comm, sigmasq_comm, P_comm, F_sensing, gamma, woa_params)
% opt_comm_WOA_BHC - Communication beamforming optimization using WOA-BHC
%
% Replaces opt_comm_SOCP_vec.m - Optimizes communication beamforming vectors
% to satisfy SINR constraints and power constraints using WOA-BHC.
%
% Inputs:
%   H_comm     - Communication channel matrix (U x M x N)
%   sigmasq_comm - Noise variance for communication
%   P_comm     - Communication power budget per AP
%   F_sensing  - Sensing beamforming matrix (T x M x N)
%   gamma      - Target SINR threshold
%   woa_params - (Optional) WOA-BHC parameters struct with fields:
%                 .SearchAgents, .MaxIter, .bias_factor, .boundary_threshold
%
% Outputs:
%   F_star     - Optimal beamforming matrix (U x M x N)
%   feasible   - Boolean indicating if a feasible solution was found

    % Extract dimensions
    [U, M, N] = size(H_comm);
    
    % Stack H_comm for efficient computation
    H_st = reshape(H_comm, U, []); % U x (M*N)
    
    % Stack sensing beamforming
    [T, ~, ~] = size(F_sensing);
    F_sensing_st = reshape(F_sensing, T, []);
    
    % Create selection matrices for per-AP power constraints
    D = zeros(M, M*N, M*N);
    for m = 1:M
        diag_idx = (m:M:M*N);
        D(m, diag_idx, diag_idx) = eye(N);
    end
    
    % Default WOA-BHC parameters
    if nargin < 6 || isempty(woa_params)
        woa_params.SearchAgents = 30;
        woa_params.MaxIter = 100;
        woa_params.bias_factor = 0.4;
        woa_params.boundary_threshold = 0.15;
    end
    
    % Decision variable dimension: F is U x (M*N) complex
    % Encode as real vector: [real(F(:)); imag(F(:))]
    dim = 2 * U * M * N;
    
    % Set bounds based on power constraint
    max_amp = sqrt(P_comm); % Maximum amplitude per element
    lb = -max_amp * ones(1, dim);
    ub = max_amp * ones(1, dim);
    
    % Define objective function with penalty method
    objective = @(x) compute_penalty(x, H_st, F_sensing_st, sigmasq_comm, P_comm, gamma, U, M, N, D);
    
    % Run WOA-BHC optimization
    [best_pos, best_score, ~] = WOA_BHC(objective, dim, woa_params.SearchAgents, ...
        woa_params.MaxIter, lb, ub, woa_params.bias_factor, woa_params.boundary_threshold);
    
    % Decode solution
    F = decode_beamforming(best_pos, U, M, N);
    
    % Check feasibility
    [penalty, sinr_violation, power_violation] = compute_penalty(best_pos, H_st, F_sensing_st, ...
        sigmasq_comm, P_comm, gamma, U, M, N, D);
    
    % Consider feasible if constraints are approximately satisfied
    feasibility_tol = 1e-3;
    if sinr_violation < feasibility_tol && power_violation < feasibility_tol
        feasible = true;
        F_star = reshape(F, U, M, N);
    else
        feasible = false;
        % Return best found solution even if not strictly feasible
        F_star = reshape(F, U, M, N);
    end
end

function [total_penalty, sinr_violation, power_violation] = compute_penalty(x, H_st, F_sensing_st, sigmasq_comm, P_comm, gamma, U, M, N, D)
    % Decode beamforming from real vector
    F = decode_beamforming(x, U, M, N);
    F_vec = reshape(F, U, M*N);
    
    % Penalty weights
    sinr_weight = 1000;
    power_weight = 1000;
    
    % SINR constraint penalties
    sinr_violation = 0;
    for u = 1:U
        h_u = H_st(u, :);
        
        % Desired signal power
        signal_power = abs(h_u * F_vec(u, :)')^2;
        
        % Interference from other users
        interference = 0;
        for k = 1:U
            if k ~= u
                interference = interference + abs(h_u * F_vec(k, :)')^2;
            end
        end
        
        % Interference from sensing
        for t = 1:size(F_sensing_st, 1)
            interference = interference + abs(h_u * F_sensing_st(t, :)')^2;
        end
        
        % Total interference + noise
        total_interference = interference + sigmasq_comm;
        
        % SINR for user u
        sinr_u = signal_power / total_interference;
        
        % Penalty for SINR constraint violation (SINR_u >= gamma)
        if sinr_u < gamma
            sinr_violation = sinr_violation + (gamma - sinr_u)^2;
        end
    end
    
    % Power constraint penalties (per AP)
    power_violation = 0;
    for m = 1:M
        Dm = squeeze(D(m, :, :));
        power_m = sum(abs(F_vec * Dm).^2, 'all');
        if power_m > P_comm
            power_violation = power_violation + (power_m - P_comm)^2;
        end
    end
    
    % Total penalty (objective to minimize)
    total_penalty = sinr_weight * sinr_violation + power_weight * power_violation;
    
    % Add small regularization to prefer lower power solutions
    total_power = sum(abs(F_vec).^2, 'all');
    total_penalty = total_penalty + 0.001 * total_power;
end

function F = decode_beamforming(x, U, M, N)
    % Decode real vector to complex beamforming matrix
    num_elements = U * M * N;
    real_part = x(1:num_elements);
    imag_part = x(num_elements+1:end);
    F_vec = real_part + 1j * imag_part;
    F = reshape(F_vec, U, M*N);
end
