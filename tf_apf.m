function [R_best,L_best,Shortest_Route,Shortest_Length] = tf_apf(D,initial,destination,dis,h,NC_max,m,MM,Lgrid)
%% 传统人工势场法路径规划函数
%% 主要参数说明
%% D        8方向距离转移矩阵（含障碍物信息）
%% initial  初始坐标
%% destination 目标坐标
%% NC_max   最大迭代次数
%% m        随机扰动尝试次数
%% MM       栅格地图尺寸
%% Lgrid    栅格边长

%% 第一步：参数初始化
NC = 1;
R_best = zeros(NC_max, MM^2);
L_best = inf.*ones(NC_max, 1);

% 人工势场参数
K_att = 0.5;        % 引力增益系数
K_rep = 0.8;        % 斥力增益系数
rho_0 = 3*Lgrid;    % 障碍影响距离
goal_threshold = Lgrid; % 目标阈值

% 坐标转换
inum = MM + (initial(1)-0.5)*MM - (initial(2)-0.5);
dnum = MM + (destination(1)-0.5)*MM - (destination(2)-0.5);
Dir = [-MM-1, -1, MM-1, MM, MM+1, 1, 1-MM, -MM];

while NC <= NC_max
    %% 第二步：单次路径搜索
    path = zeros(1, MM^2);
    dir_sequence = zeros(1, MM^2);
    current_node = inum;
    path(1) = current_node;
    step = 1;
    stuck = false;
    visited = false(1, MM^2);
    
    while current_node ~= dnum && ~stuck && step < MM^2
        visited(current_node) = true;
        
        % 计算目标引力
        [xg, yg] = ind2sub([MM, MM], dnum);
        [x, y] = ind2sub([MM, MM], current_node);
        F_att = K_att * sqrt((xg-x)^2 + (yg-y)^2);
        
        % 获取可行方向
        feasible_dirs = find(D(current_node,:) ~= inf);
        next_nodes = current_node + Dir(feasible_dirs);
        
        % 排除越界和已访问节点
        valid_range = (next_nodes > 0) & (next_nodes <= MM^2);
        feasible_dirs_range = feasible_dirs(valid_range);
        next_nodes_range = next_nodes(valid_range);
        
        valid_visited = ~visited(next_nodes_range); % 注意移除转置符
        feasible_dirs = feasible_dirs_range(valid_visited);
        next_nodes = next_nodes_range(valid_visited);
        
        if isempty(next_nodes)
            stuck = true;
            break;
        end
        
        % 计算各方向势能
        potentials = zeros(1, length(next_nodes));
        for k = 1:length(next_nodes)
            % 目标引力
            [xn, yn] = ind2sub([MM, MM], next_nodes(k));
            att_force = K_att * sqrt((xg-xn)^2 + (yg-yn)^2);
            
            % 障碍斥力
            rep_force = 0;
            for d = 1:8
                if D(next_nodes(k),d) == inf
                    [xo, yo] = ind2sub([MM, MM], next_nodes(k)+Dir(d));
                    rho = sqrt((xn-xo)^2 + (yn-yo)^2);
                    if rho < rho_0
                        rep_force = rep_force + K_rep*(1/rho - 1/rho_0)*(1/rho^2);
                    end
                end
            end
            
            potentials(k) = att_force + rep_force;
        end
        
        % 选择最小势能方向
        [~, min_idx] = min(potentials);
        chosen_dir = feasible_dirs(min_idx);
        current_node = next_nodes(min_idx);
        
        step = step + 1;
        path(step) = current_node;
        dir_sequence(step-1) = chosen_dir;
        
        % 检查是否陷入局部极小
        if step > 3 && path(step) == path(step-2)
            stuck = true;
        end
    end
    
    %% 第三步：记录有效路径
    if ~stuck && current_node == dnum
        path_length = sum(D(sub2ind(size(D), path(1:step-1), dir_sequence(1:step-1))))*Lgrid;
        
        if path_length < L_best(NC)
            L_best(NC) = path_length;
            R_best(NC,1:step) = path(1:step);
        end
    end
    
    NC = NC + 1;
end

%% 第四步：提取最优路径
valid_routes = R_best(L_best < inf,:);
if ~isempty(valid_routes)
    [Shortest_Length, idx] = min(L_best);
    Shortest_Route = R_best(idx,:);
    Shortest_Route = Shortest_Route(Shortest_Route ~= 0);
else
    Shortest_Length = inf;
    Shortest_Route = [];
end
end