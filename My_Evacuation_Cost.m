function cost = My_Safe_Fitness(x, starts, refuges)
    % 1. 强制取整
    idx_list = round(x);
    
    % 2. 处理非正常数值 (NaN/Inf)
    idx_list(isnan(idx_list)) = 1;
    idx_list(isinf(idx_list)) = 1;
    
    num_pts = size(starts, 1);
    num_ref = size(refuges, 1);
    
    % 3. 严格范围截断，防止报错
    idx_list(idx_list < 1) = 1;
    idx_list(idx_list > num_ref) = num_ref;
    
    total_dist = 0;
    
    % 4. 计算距离代价
    for i = 1:num_pts
        target = idx_list(i);
        % 计算欧氏距离平方
        d = (starts(i,1) - refuges(target,1))^2 + (starts(i,2) - refuges(target,2))^2;
        total_dist = total_dist + d;
    end
    
    % 5. 均衡性惩罚
    counts = histcounts(idx_list, 1:num_ref+1);
    penalty = sum(max(0, counts - (num_pts/num_ref)*2)) * 1000;
    
    cost = total_dist + penalty;
end