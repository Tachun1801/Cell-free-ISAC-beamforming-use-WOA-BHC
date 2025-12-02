function [F_star, feasible, SSNR_opt] = opt_jsc_WOA_BHC(H_comm, sigmasq_comm, gamma, sensing_beamsteering, sensing_streams, sigmasq_sens, P_all, woa_params)
% opt_jsc_WOA_BHC - Joint Sensing-Communication optimization using WOA-BHC
%
% Replaces opt_jsc_SDP.m - Maximizes sensing SNR while maintaining 
% SINR >= gamma and power constraints using WOA-BHC.
%
% Inputs:
%   H_comm              - Communication channel matrix (U x M x N)
%   sigmasq_comm        - Noise variance for communication
%   gamma               - Target SINR threshold
%   sensing_beamsteering - Sensing beamsteering vector (T x M x N)
%   sensing_streams     - Number of sensing streams
%   sigmasq_sens        - Sensing channel variance (radar RCS)
%   P_all               - Total power budget per AP
%   woa_params          - (Optional) WOA-BHC parameters struct
%
% Outputs:
%   F_star    - Joint beamforming matrix (M*N x M*N x num_streams)
%   feasible  - Boolean indicating feasibility
%   SSNR_opt  - Optimal sensing SNR achieved

    % Extract dimensions
    [U, M, N] = size(H_comm);
    H_st = reshape(H_comm, U, []); % U x (M*N)
    
    % Number of total streams (communication + sensing)
    num_streams = U + sensing_streams;
    
    % Create selection matrices for per-AP power constraints
    D = zeros(M, M*N, M*N);
    for m = 1:M
        diag_idx = (m:M:M*N);
        D(m, diag_idx, diag_idx) = eye(N);
    end
    
    % Sensing beamsteering vector
    a = reshape(sensing_beamsteering, 1, []);
    
    % Default WOA-BHC parameters
    if nargin < 8 || isempty(woa_params)
        woa_params.SearchAgents = 30;
        woa_params.MaxIter = 100;
        woa_params.bias_factor = 0.4;
        woa_params.boundary_threshold = 0.15;
    end
    
    % Decision variable dimension: beamforming vectors for all streams
    % Each stream has M*N complex elements
    dim = 2 * num_streams * M * N;
    
    % Set bounds
    max_amp = sqrt(P_all);
    lb = -max_amp * ones(1, dim);
    ub = max_amp * ones(1, dim);
    
    % Define objective function (negative because WOA minimizes)
    objective = @(x) compute_jsc_objective(x, H_st, a, sigmasq_comm, sigmasq_sens, ...
        gamma, P_all, U, M, N, D, num_streams);
    
    % Run WOA-BHC optimization
    [best_pos, ~, ~] = WOA_BHC(objective, dim, woa_params.SearchAgents, ...
        woa_params.MaxIter, lb, ub, woa_params.bias_factor, woa_params.boundary_threshold);
    
    % Decode solution to get beamforming matrices
    [F_streams, F_matrices] = decode_jsc_beamforming(best_pos, M, N, num_streams);
    
    % Check feasibility
    [~, sinr_violation, power_violation, sensing_snr] = compute_jsc_objective(best_pos, H_st, a, ...
        sigmasq_comm, sigmasq_sens, gamma, P_all, U, M, N, D, num_streams);
    
    feasibility_tol = 1e-2;
    if sinr_violation < feasibility_tol && power_violation < feasibility_tol
        feasible = true;
    else
        feasible = false;
    end
    
    % Output in same format as opt_jsc_SDP: (M*N x M*N x num_streams)
    F_star = F_matrices;
    SSNR_opt = sensing_snr;
end

function [total_obj, sinr_violation, power_violation, sensing_snr] = compute_jsc_objective(x, H_st, a, sigmasq_comm, sigmasq_sens, gamma, P_all, U, M, N, D, num_streams)
    % Decode beamforming vectors
    [F_streams, ~] = decode_jsc_beamforming(x, M, N, num_streams);
    
    % Penalty weights
    sinr_weight = 1000;
    power_weight = 1000;
    
    % Calculate sensing SNR (objective to maximize)
    sensing_gain = 0;
    for s = 1:num_streams
        f_s = F_streams(s, :);
        for m = 1:M
            Dm = squeeze(D(m, :, :));
            a_m = a * Dm;
            gain_m = abs(a_m * f_s')^2;
            sensing_gain = sensing_gain + gain_m;
        end
    end
    sensing_snr = sensing_gain * sigmasq_sens;
    
    % SINR constraint penalties (for communication users)
    sinr_violation = 0;
    for u = 1:U
        h_u = H_st(u, :);
        f_u = F_streams(u, :);
        
        % Desired signal power
        signal_power = abs(h_u * f_u')^2;
        
        % Interference from other streams
        interference = 0;
        for k = 1:num_streams
            if k ~= u
                interference = interference + abs(h_u * F_streams(k, :)')^2;
            end
        end
        
        % Total interference + noise
        total_interference = interference + sigmasq_comm;
        
        % SINR for user u
        if total_interference > 0
            sinr_u = signal_power / total_interference;
        else
            sinr_u = inf;
        end
        
        % Penalty for SINR constraint violation (SINR_u >= gamma)
        if sinr_u < gamma
            sinr_violation = sinr_violation + (gamma - sinr_u)^2;
        end
    end
    
    % Power constraint penalties (per AP)
    power_violation = 0;
    for m = 1:M
        Dm = squeeze(D(m, :, :));
        power_m = 0;
        for s = 1:num_streams
            power_m = power_m + sum(abs(F_streams(s, :) * Dm).^2);
        end
        if power_m > P_all
            power_violation = power_violation + (power_m - P_all)^2;
        end
    end
    
    % Total objective: minimize (-sensing_snr + penalties)
    % WOA minimizes, so negate sensing_snr to maximize it
    total_obj = -sensing_snr + sinr_weight * sinr_violation + power_weight * power_violation;
end

function [F_streams, F_matrices] = decode_jsc_beamforming(x, M, N, num_streams)
    % Decode real vector to complex beamforming vectors and matrices
    MN = M * N;
    elements_per_stream = MN;
    total_elements = num_streams * elements_per_stream;
    
    real_part = x(1:total_elements);
    imag_part = x(total_elements+1:end);
    
    F_streams = zeros(num_streams, MN);
    for s = 1:num_streams
        idx_start = (s-1) * elements_per_stream + 1;
        idx_end = s * elements_per_stream;
        F_streams(s, :) = real_part(idx_start:idx_end) + 1j * imag_part(idx_start:idx_end);
    end
    
    % Also create the matrix format (M*N x M*N x num_streams) for compatibility
    F_matrices = zeros(MN, MN, num_streams);
    for s = 1:num_streams
        f_s = F_streams(s, :).';
        F_matrices(:, :, s) = f_s * f_s';
    end
end
