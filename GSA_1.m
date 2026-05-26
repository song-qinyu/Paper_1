clear; clc; close all;

%% 1. 加载数据
load('sj5.mat'); 
starts = data.start;    
binan_raw = data.binan;  
num_points = size(starts, 1);
num_refuges = size(binan_raw, 1);

%% 2. 空间约束预处理 (确保不乱飞的关键)
K = 5; % 每个点只在最近的 5 个避难所中选
fprintf('正在计算局部最优候选范围...\n');
candidate_matrix = zeros(num_points, K);
for i = 1:num_points
    % 计算该点到所有避难所的平方距离
    dists = sum((starts(i,:) - binan_raw).^2, 2);
    [~, sorted_idx] = sort(dists);
    candidate_matrix(i, :) = sorted_idx(1:K); % 第一列即为“最近”的避难所
end

%% 3. GSA 参数设置
dim = num_points;     
lb = 1; ub = K;  % 搜索空间压缩为 1~K
pop = 30;     
maxIter = 100;

% 适应度函数
fobj = @(x) My_Spatial_Fitness(x, starts, binan_raw, candidate_matrix);

fprintf('开始全量分配优化（带引导机制）...\n');
% 运行 GSA
[Best_pos, Best_fitness, Iter_curve, ~, ~] = GSA(pop, maxIter, lb, ub, dim, fobj);

%% 4. 结果转换：从局部索引映射回 1~59 编号
final_local_idx = max(1, min(K, round(Best_pos)));
final_alloc = zeros(1, num_points);
for i = 1:num_points
    final_alloc(i) = candidate_matrix(i, final_local_idx(i));
end

%% 5. 全量绘图展示 (解决线不全、不精准的问题)
figure(1); clf; 
set(gcf, 'Color', 'w', 'Name', '疏散分配方案');
hold on;

% 绘制底图道路
try
    road = shaperead('E:\论文格式\NSGA\road_2.shp'); 
    for i = 1:length(road)
        plot(road(i).X, road(i).Y, 'Color', [0.85 0.85 0.85], 'LineWidth', 0.5); 
    end
catch
end

% A. 先画线 (确保点在上面)
colors = lines(num_refuges);
fprintf('正在渲染 2674 条分配线...\n');
for i = 1:num_points
    ref_idx = final_alloc(i);
    % 使用极细线条和透明度
    line([starts(i,1), binan_raw(ref_idx,1)], [starts(i,2), binan_raw(ref_idx,2)], ...
         'Color', [colors(ref_idx,:), 0.25], 'LineWidth', 0.5);
end

% B. 再画点 (覆盖线头，视觉更精准)
plot(starts(:,1), starts(:,2), 'r.', 'MarkerSize', 2);
plot(binan_raw(:,1), binan_raw(:,2), 'g^', 'MarkerFaceColor','g','MarkerSize',8);

title(['全量分配结果 (K=', num2str(K), ' 局部约束)']);
axis tight; axis equal; box on;

% 绘制收敛曲线
figure(2);
plot(Iter_curve, 'b', 'LineWidth', 2);
grid on;
title('GSA 收敛曲线'); xlabel('迭代次数'); ylabel('距离代价');