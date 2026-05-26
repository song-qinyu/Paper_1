function cost = My_Final_Capacity_Fitness(x, starts, refuges, capacities)
    % 1. 索引处理
    idx_list = round(x);
    num_ref = size(refuges, 1);
    idx_list(isnan(idx_list)) = 1;
    idx_list(idx_list < 1) = 1;
    idx_list(idx_list > num_ref) = num_ref;
    
    % 2. 距离代价：使用矩阵向量化运算加速
    % 计算分配后的对应避难所坐标
    assigned_refuges = refuges(idx_list, :);
    % 计算所有点到其分配目标的欧氏距离平方
    % 使用平方和能极大地抑制“长线”生成，让分配更精准
    dist_sq = sum((starts - assigned_refuges).^2, 2);
    total_dist = sum(dist_sq);
    
    % 3. 容量惩罚
    penalty = 0;
    counts = histcounts(idx_list, 1:num_ref+1);
    
    % 超载惩罚逻辑
    diff = counts - capacities;
    overload = sum(diff(diff > 0)); % 只对超出容量的部分进行惩罚
    
    % 惩罚系数调大，强制让 GSA 寻找满足容量的解
    penalty = overload * 10^5; 
    
    cost = total_dist + penalty;
end