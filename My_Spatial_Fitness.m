function cost = My_Spatial_Fitness(x, starts, refuges, candidate_matrix)
    % 1. 局部索引提取
    local_idx_raw = x; 
    local_idx = round(local_idx_raw);
    num_pts = size(starts, 1);
    K = size(candidate_matrix, 2);
    
    % 索引保护
    local_idx(local_idx < 1) = 1;
    local_idx(local_idx > K) = K;
    
    % 2. 映射全局编号
    global_idx = zeros(num_pts, 1);
    for i = 1:num_pts
        global_idx(i) = candidate_matrix(i, local_idx(i));
    end
    
    % 3. 基础距离代价 (向量化加速)
    assigned_ref = refuges(global_idx, :);
    dist_sq = sum((starts - assigned_ref).^2, 2);
    base_cost = sum(dist_sq);
    
    % 4. 引导项：给更近的候选者极微小的奖励
    % 这样即使随机生成的粒子，也会产生朝向“最近点”移动的趋势
    guide_penalty = sum(local_idx_raw) * 0.1; 
    
    cost = base_cost + guide_penalty;
end