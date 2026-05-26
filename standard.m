function [R_best,F_best,L_best,T_best,L_ave,Shortest_Route,Shortest_Length] = standard(D,initial,destination,dis,h,NC_max,m, ...
    t,Rho,Omega,Mu,Lambda,Q,G,MM,Lgrid,Eta_omega,Eta_obs,Eta_h,Uatt,Eta_hobs)

% 初始化返回值
R_best = []; F_best = 0; L_best = inf*ones(NC_max,1); T_best = 0; 
L_ave = 0; Shortest_Route = []; Shortest_Length = inf;

% 索引转换
try
    inum = sub2ind([MM, MM], round(initial(2)), round(initial(1)));
    dnum = sub2ind([MM, MM], round(destination(2)), round(destination(1)));
catch
    return;
end

Dir = [-MM-1, -1, MM-1, MM, MM+1, 1, 1-MM, -MM];
Tau = ones(MM^2,8); 
Eta = 1./(10^(-5) + Uatt); 

NC = 1;
R_record = zeros(NC_max, MM^2);

while NC <= NC_max
    Alpha = (NC_max/(10*NC))+1; Beta = (3*NC/NC_max)+1;
    Tabu = zeros(m, MM^2); to_direct = zeros(m, MM^2);
    Tabu(:,1) = inum;
    
    for i = 1:m
        j = 2;
        while Tabu(i,j-1) ~= dnum
            visited = Tabu(i, 1:(j-1));
            curr = visited(end);
            J = []; N = []; Jc = 1;
            for k = 1:8
                k1 = curr + Dir(k);
                if k1 <= 0 || k1 > MM^2 || D(curr, k) == inf, continue; end
                if ~any(visited == k1)
                    J(Jc) = k1; N(Jc) = k; Jc = Jc+1;
                end
            end
            if isempty(J), Tabu(i,:) = 0; break; end
            
            % 转移概率计算
            Pz = (Tau(curr, N).^Alpha) .* (Eta(curr, N).^Beta) .* (1./(dis(J)+eps)).^Beta;
            Pz = Pz / sum(Pz);
            Select = find(cumsum(Pz) >= rand, 1);
            if isempty(Select), break; end
            to_direct(i, j-1) = N(Select);
            Tabu(i, j) = J(Select);
            j = j + 1;
            if j > MM^2, break; end
        end
    end
    
    % 评估
    L = inf * ones(m, 1);
    for i = 1:m
        if Tabu(i, 1) ~= 0 && any(Tabu(i, :) == dnum)
            route = Tabu(i, Tabu(i, :) > 0);
            p_len = 0;
            for r = 1:(length(route)-1)
                p_len = p_len + D(route(r), to_direct(i, r));
            end
            L(i) = p_len;
        end
    end
    
    if ~all(L == inf)
        [L_best(NC), pos] = min(L);
        R_record(NC, :) = Tabu(pos(1), :);
        % 更新信息素
        for i = 1:m
            if L(i) < inf
                for step = 1:MM^2-1
                    if Tabu(i, step) == 0 || Tabu(i, step+1) == 0, break; end
                    Tau(Tabu(i,step), to_direct(i,step)) = Tau(Tabu(i,step), to_direct(i,step)) + Q/L(i);
                end
            end
        end
    end
    Tau = (1-Rho) * Tau;
    NC = NC + 1;
end

[Shortest_Length, idx] = min(L_best);
if Shortest_Length < inf
    Shortest_Route = R_record(idx, :);
    Shortest_Route = Shortest_Route(Shortest_Route > 0);
end
end