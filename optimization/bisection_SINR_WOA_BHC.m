function [F_star_all, feasible_SINR] = bisection_SINR_WOA_BHC(low, high, tol, fun)
% bisection_SINR_WOA_BHC - Find maximum achievable SINR using bisection search
%
% Replaces bisection_SINR.m - Uses bisection search to find the maximum
% SINR value that can be achieved with feasible beamforming solution.
%
% Inputs:
%   low   - Lower bound for SINR search
%   high  - Upper bound for SINR search  
%   tol   - Tolerance for bisection convergence
%   fun   - Function handle that takes gamma and returns [F, feasible]
%           (e.g., @(gamma) opt_comm_WOA_BHC(..., gamma))
%
% Outputs:
%   F_star_all    - Beamforming matrix corresponding to max achievable SINR
%   feasible_SINR - Maximum achievable SINR value

    F_star_all = [];
    feasible_SINR = [];
    
    while (high - low) > tol
        mid = (high + low) / 2;
        [F_star, feasible] = fun(mid);
        
        if feasible
            low = mid;
            F_star_all = F_star;
            feasible_SINR = mid;
        else
            high = mid;
            if isempty(F_star_all)
                F_star_all = F_star;
            end
        end
    end
    
    % If no feasible solution found, return the last attempted solution
    if isempty(feasible_SINR)
        feasible_SINR = low;
    end
end
