function [F_star_comm, F_star_sensing] = WOA_BHC_beam_extraction(F_star, H_comm)
% WOA_BHC_beam_extraction - Extract beamforming vectors from joint optimization result
%
% Replaces SDP_beam_extraction.m - Extracts communication and sensing 
% beamforming vectors from the joint beamforming matrix.
%
% Inputs:
%   F_star    - Joint beamforming matrix (M*N x M*N x num_streams) from opt_jsc_WOA_BHC
%   H_comm    - Communication channel matrix (U x M x N)
%
% Outputs:
%   F_star_comm    - Communication beamforming matrix (U x M x N)
%   F_star_sensing - Sensing beamforming matrix (S x M x N)

    % Extract dimensions
    [U, M, N] = size(H_comm);
    S = size(F_star, 3); % Total number of streams
    
    F_star_comm = zeros(U, M, N);
    H_comm_stacked = reshape(H_comm, U, []); % U x (M*N)
    
    F_comm_sum = 0;
    
    % Extract communication beamforming vectors (first U streams)
    for u = 1:U
        Q_u = F_star(:, :, u).';
        h_u = H_comm_stacked(u, :).';
        
        % Rank-1 approximation: extract principal eigenvector
        denom = h_u' * Q_u * h_u;
        if abs(denom) > 1e-10
            f_u = denom^(-1/2) * Q_u * h_u;
        else
            % Fallback: use eigenvector decomposition
            [V, D] = eig(Q_u);
            [~, max_idx] = max(diag(D));
            f_u = V(:, max_idx);
            f_u = f_u / norm(f_u); % Normalize
        end
        
        F_star_u = reshape(f_u, M, N);
        F_star_comm(u, :, :) = F_star_u;
        
        F_u_hat = f_u * f_u';
        F_comm_sum = F_comm_sum + F_u_hat;
    end
    
    % Extract sensing beamforming vectors (remaining S-U streams)
    num_sensing_streams = max(S - U, 1);
    F_star_sensing = zeros(num_sensing_streams, M, N);
    
    if S > U
        F_star_sum = sum(F_star, 3).';
        F_sens_sum = F_star_sum - F_comm_sum;
        
        % Eigenvalue decomposition to extract sensing beams
        [eigenvec, D] = eig(F_sens_sum);
        [lambda, order] = sort(diag(D), 'descend');
        
        for s = U+1:S
            s_idx = s - U;
            if lambda(s_idx) > 0
                F_star_sensing(s_idx, :, :) = reshape(sqrt(lambda(s_idx)) * eigenvec(:, order(s_idx)), 1, M, N);
            end
        end
    end
end
