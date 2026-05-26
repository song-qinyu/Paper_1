function cost = My_Distance_Only_Fitness(x, starts, refuges)
    % 1. 索引处理
    idx_list = round(x);
    num_ref = size(refuges, 1);
    idx_list(isnan(idx_list)) = 1;
    idx_list(idx_list < 1) = 1;
    idx_list(idx_list > num_ref) = num_ref;
    
    % 2. 向量化计算距离 (速度快，有助于收敛)
    % 直接提取所有分配到的避难所坐标
    assigned_ref = refuges(idx_list, :);
    
    % 计算平方距离之和
    % 在高维度下，平方距离能让算法更敏感，从而更容易产生收敛坡度
    dist_sq = sum((starts - assigned_ref).^2, 2);
    cost = sum(dist_sq);
end