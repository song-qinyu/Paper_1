function cost = My_Safe_Fitness_Full(x, starts, refuges, cap_data)
    % 1. 索引处理
    idx_list = round(x);
    idx_list(isnan(idx_list)) = 1;
    idx_list(idx_list < 1) = 1;
    idx_list(idx_list > size(refuges, 1)) = size(refuges, 1);
    
    % 2. 距离代价（使用矩阵运算，2674个点也不卡）
    % 找到每个点分配到的避难所坐标
    assigned_ref = refuges(idx_list, :);
    % 计算平方距离之和 (d^2)
    dist_sq = sum((starts - assigned_ref).^2, 2);
    total_dist = sum(dist_sq);
    
    % 3. 容量惩罚（基于 YFenPei）
    counts = histcounts(idx_list, 1:size(refuges, 1)+1);
    % 找出超出容量的部分
    overload = sum(max(0, counts - cap_data));
    
    % 总代价 = 距离 + 极高权重的容量惩罚
    % 增加一个 1e5 的系数，让算法在保证“距离近”的同时，必须先解决“装不下”的问题
    cost = total_dist + overload * 100000;
end