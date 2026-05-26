clc; clear; close all; tic;

%% 1. 数据加载与预处理
load('sj5.mat'); 
starts = data.start;    
binan_raw = data.binan;  
num_points = size(starts, 1);
num_refuges = size(binan_raw, 1);

% 统一坐标偏移 (关键：确保所有图层对齐)
off_x = min(starts(:,1)); off_y = min(starts(:,2));
hx = starts(:,1) - off_x; hy = starts(:,2) - off_y;
all_sx = binan_raw(:,1) - off_x; all_sy = binan_raw(:,2) - off_y;

%% 2. 空间约束预处理
K = 5; 
fprintf('正在计算局部最优候选范围...\n');
candidate_matrix = zeros(num_points, K);
for i = 1:num_points
    dists = sum((starts(i,:) - binan_raw).^2, 2);
    [~, sorted_idx] = sort(dists);
    candidate_matrix(i, :) = sorted_idx(1:K); 
end

%% 3. GSA 优化过程 (带跳出死锁机制)
dim = num_points;     
lb = 1; ub = K;  
pop = 30;     
maxIter = 100;
G0 = 100; % 初始引力常数

% 初始化粒子
X = lb + (ub - lb) * rand(pop, dim);
V = zeros(pop, dim);
Best_fitness = inf;
Iter_curve = zeros(maxIter, 1);
no_improve_count = 0;

fprintf('GSA 优化启动 (已开启防死锁扰动)...\n');

for t = 1:maxIter
    % 1. 计算适应度
    fitness = zeros(pop, 1);
    for i = 1:pop
        fitness(i) = My_Spatial_Fitness(X(i,:), starts, binan_raw, candidate_matrix);
    end
    
    % 2. 更新全局最优
    [min_f, best_idx] = min(fitness);
    if min_f < Best_fitness
        Best_fitness = min_f;
        Best_pos = X(best_idx, :);
        no_improve_count = 0;
    else
        no_improve_count = no_improve_count + 1;
    end
    
    % --- 【核心改进：死锁跳出机制】 ---
    if no_improve_count > 5
        % 随机挑选一半粒子进行突变，强制跨越四舍五入的阈值
        mutation_mask = rand(pop, dim) < 0.2; 
        X(mutation_mask) = lb + (ub - lb) * rand(sum(mutation_mask(:)), 1);
        no_improve_count = 0;
        fprintf('Iter %d: 检测到死锁，触发强制扰动...\n', t);
    end
    
    % 3. 计算质量 M
    best_f = min(fitness); worst_f = max(fitness);
    if best_f == worst_f
        M = ones(pop, 1);
    else
        M = (fitness - worst_f) ./ (best_f - worst_f);
    end
    M = M ./ sum(M);
    
    % 4. 更新引力常数 G 和加速度 a
    G = G0 * exp(-20 * t / maxIter);
    a = zeros(pop, dim);
    for i = 1:pop
        for j = 1:pop
            if i ~= j
                R = norm(X(i,:) - X(j,:), 2) + eps;
                a(i,:) = a(i,:) + rand * G * M(j) * (X(j,:) - X(i,:)) / R;
            end
        end
    end
    
    % 5. 更新速度和位置
    V = rand * V + a;
    X = X + V;
    
    % 边界处理
    X = max(min(X, ub), lb);
    Iter_curve(t) = Best_fitness;
    
    if mod(t, 10) == 0
        fprintf('迭代: %d, 最佳距离: %.2e\n', t, Best_fitness);
    end
end

% 结果解析：映射回原始避难所索引
final_local_idx = max(1, min(K, round(Best_pos)));
final_alloc = zeros(1, num_points);
for i = 1:num_points
    final_alloc(i) = candidate_matrix(i, final_local_idx(i));
end

%% 4. 三图标准模板输出 (统一视觉风格)

% --- 图 1: 2D 分配方案图 ---
figure('Color','w', 'Name', 'GSA 2D Allocation'); hold on; box on;
if isfield(data, 'road')
    for k = 1:length(data.road)
        plot(data.road{k}(:,1)-off_x, data.road{k}(:,2)-off_y, 'Color', [0.85 0.85 0.85], 'LineWidth', 0.5);
    end
end
for i = 1:num_points
    ref_idx = final_alloc(i);
    line([hx(i), all_sx(ref_idx)], [hy(i), all_sy(ref_idx)], 'Color', [1.0, 0.0, 0.0, 0.15]);
end
h_res = scatter(hx, hy, 10, [0.0, 0.2, 0.6], 'filled');
h_shl = scatter(all_sx, all_sy, 70, 'g', '^', 'filled', 'MarkerEdgeColor', 'k');

xlabel('X Coordinate Offset (m)', 'FontWeight', 'bold');
ylabel('Y Coordinate Offset (m)', 'FontWeight', 'bold');
ax = gca; ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0; 
axis equal; grid on; title('GSA Algorithm: Shelter Allocation (2D)');
legend([h_shl, h_res], {'Optimized Shelter', 'Residential Area'}, 'Location', 'northeast');

% --- 图 2: 收敛过程图 ---
figure('Color','w', 'Name', 'GSA Convergence');
plot(Iter_curve, 'LineWidth', 2, 'Color', [0.85, 0.33, 0.1]);
grid on; xlabel('Iteration'); ylabel('Best Fitness (Total Distance)');
title('GSA Optimization Convergence Curve');

% --- 图 3: 3D 需求地形图 (不含宅基地) ---
figure('Color','w', 'Name', 'GSA 3D View'); hold on; grid on;
res = 80; [Xq, Yq] = meshgrid(linspace(min(hx), max(hx), res), linspace(min(hy), max(hy), res));
Z = zeros(size(Xq)); bw = (max(hx)-min(hx))/25;
sample_idx = randperm(length(hx), min(1000, length(hx)));
for i = sample_idx
    Z = Z + exp(-((Xq-hx(i)).^2 + (Yq-hy(i)).^2) / (2*bw^2));
end
surf(Xq, Yq, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.6); 
colormap(jet); shading interp; 
cb = colorbar; ylabel(cb, 'Demand Intensity');

z_top = max(Z(:)) * 1.2;
for k = 1:num_refuges
    plot3([all_sx(k), all_sx(k)], [all_sy(k), all_sy(k)], [0, z_top], 'r--', 'LineWidth', 0.8);
end
h_3d = scatter3(all_sx, all_sy, ones(1, num_refuges)*z_top, 80, 'g', '^', 'filled', 'MarkerEdgeColor', 'k');

xlabel('X Offset (m)'); ylabel('Y Offset (m)'); zlabel('Intensity');
ax3 = gca; ax3.XAxis.Exponent = 0; ax3.YAxis.Exponent = 0; 
view(-35, 45); axis tight; title('GSA Optimized Shelters on Demand Landscape (3D)');
legend(h_3d, 'Optimized Shelter (Air)', 'Location', 'northeast');
toc;

%% --- 内部评估函数 ---
function fitness = My_Spatial_Fitness(x, starts, binan_raw, candidate_matrix)
    indices = max(1, min(size(candidate_matrix, 2), round(x)));
    total_dist = 0;
    for i = 1:size(starts, 1)
        target_idx = candidate_matrix(i, indices(i));
        d = sqrt(sum((starts(i,:) - binan_raw(target_idx,:)).^2));
        total_dist = total_dist + d;
    end
    fitness = total_dist;
end