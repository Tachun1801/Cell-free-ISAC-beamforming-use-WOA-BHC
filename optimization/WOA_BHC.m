function [Best_pos, Best_score, Convergence_curve] = WOA_BHC(objective_func, dim, SearchAgents_no, Max_iter, lb, ub, bias_factor, boundary_threshold)
% WOA_BHC - Whale Optimization Algorithm with Biased/Boundary Hunting Component
%
% Implementation of WOA-BHC for optimization problems
% 
% Inputs:
%   objective_func     - Handle to the objective function to minimize
%   dim               - Number of dimensions (variables)
%   SearchAgents_no   - Number of search agents (whales)
%   Max_iter          - Maximum number of iterations
%   lb                - Lower bound (1 x dim vector or scalar)
%   ub                - Upper bound (1 x dim vector or scalar)
%   bias_factor       - Biased hunting factor (default: 0.4)
%   boundary_threshold - Boundary hunting threshold (default: 0.15)
%
% Outputs:
%   Best_pos          - Position of the best solution found
%   Best_score        - Best objective function value
%   Convergence_curve - History of best scores over iterations

    % Default parameters for BHC
    if nargin < 7 || isempty(bias_factor)
        bias_factor = 0.4;
    end
    if nargin < 8 || isempty(boundary_threshold)
        boundary_threshold = 0.15;
    end

    % WOA parameters
    b = 1; % Constant for logarithmic spiral
    
    % Ensure lb and ub are row vectors of length dim
    if isscalar(lb)
        lb = lb * ones(1, dim);
    end
    if isscalar(ub)
        ub = ub * ones(1, dim);
    end
    lb = lb(:)';
    ub = ub(:)';
    
    % Initialize the positions of search agents
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    
    % Initialize convergence curve
    Convergence_curve = zeros(1, Max_iter);
    
    % Calculate fitness for each search agent
    Fitness = zeros(SearchAgents_no, 1);
    for i = 1:SearchAgents_no
        Fitness(i) = objective_func(Positions(i, :));
    end
    
    % Find the best search agent
    [Best_score, idx] = min(Fitness);
    Best_pos = Positions(idx, :);
    
    % Main loop
    for iter = 1:Max_iter
        % a decreases linearly from 2 to 0
        a = 2 - iter * (2 / Max_iter);
        a2 = -1 + iter * (-1 / Max_iter); % a2 linearly decreases from -1 to -2
        
        % Update the position of each search agent
        for i = 1:SearchAgents_no
            r1 = rand(); % Random number in [0, 1]
            r2 = rand(); % Random number in [0, 1]
            
            A = 2 * a * r1 - a; % Eq. (2.3)
            C = 2 * r2;         % Eq. (2.4)
            
            l = (a2 - 1) * rand() + 1; % Random number in [-1, 1]
            p = rand();                 % Random number in [0, 1]
            
            if p < 0.5
                if abs(A) < 1
                    % Shrinking encircling mechanism (exploitation)
                    D = abs(C * Best_pos - Positions(i, :)); % Eq. (2.1)
                    Positions(i, :) = Best_pos - A * D;       % Eq. (2.2)
                else
                    % Search for prey (exploration)
                    rand_leader_idx = randi(SearchAgents_no);
                    X_rand = Positions(rand_leader_idx, :);
                    D = abs(C * X_rand - Positions(i, :));
                    Positions(i, :) = X_rand - A * D;
                end
            else
                % Spiral updating position (exploitation)
                D_Leader = abs(Best_pos - Positions(i, :));
                Positions(i, :) = D_Leader .* exp(b * l) .* cos(2 * pi * l) + Best_pos;
            end
            
            % BIASED HUNTING COMPONENT (BHC)
            % Apply biased movement towards best position with probability bias_factor
            if rand() < bias_factor
                % Move towards best position with some randomness
                Positions(i, :) = Positions(i, :) + bias_factor * rand() * (Best_pos - Positions(i, :));
            end
            
            % BOUNDARY HUNTING COMPONENT (BHC)
            % If whale is near boundary, apply special search strategy
            range = ub - lb;
            dist_to_lb = (Positions(i, :) - lb) ./ range;
            dist_to_ub = (ub - Positions(i, :)) ./ range;
            
            % Check if any dimension is near boundary
            near_lb = dist_to_lb < boundary_threshold;
            near_ub = dist_to_ub < boundary_threshold;
            
            if any(near_lb) || any(near_ub)
                % Apply boundary hunting: explore along the boundary
                for d = 1:dim
                    if near_lb(d)
                        % Explore near lower boundary
                        Positions(i, d) = lb(d) + boundary_threshold * range(d) * rand();
                    elseif near_ub(d)
                        % Explore near upper boundary
                        Positions(i, d) = ub(d) - boundary_threshold * range(d) * rand();
                    end
                end
            end
            
            % Boundary handling: Clip positions to [lb, ub]
            Positions(i, :) = max(min(Positions(i, :), ub), lb);
        end
        
        % Evaluate fitness for all agents
        for i = 1:SearchAgents_no
            Fitness(i) = objective_func(Positions(i, :));
            
            % Update best position if better solution found
            if Fitness(i) < Best_score
                Best_score = Fitness(i);
                Best_pos = Positions(i, :);
            end
        end
        
        % Store convergence data
        Convergence_curve(iter) = Best_score;
    end
end
