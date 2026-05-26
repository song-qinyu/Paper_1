%% DE 疏散分配优化 - 统一 3D 增强版
clc; clear; close all; tic;

% ========================== 1. 加载与适配数据 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat');
end

% 坐标偏移处理 (统一绘图基准)
raw_x = data.start(:,1); raw_y = data.start(:,2);
offset_x = min(raw_x); offset_y = min(raw_y);
house_x = raw_x - offset_x;
house_y = raw_y - offset_y;
binan_x = data.binan(:,1) - offset_x;
binan_y = data.binan(:,2) - offset_y;

% 适配字段
if exist('dis','var'), data.dis = dis; end
data.P = cellfun(@(x) length(x)-1, data.DFenPei);

% 预处理所有点的分配状态
numTotalHomes = size(data.start, 1);
data.YFenPei = zeros(1, size(data.binan, 1));
data.alldis = 0;
data.FID = []; % 存储固定分配点
for k = 1:length(B)
    if length(B{k}) == 1
        targetBinan = B{k};
        data.YFenPei(targetBinan) = data.YFenPei(targetBinan) + 12; 
        data.FID = [data.FID; k, targetBinan];
        data.alldis = data.alldis + data.dis(k, targetBinan);
    end
end

% ========================== 2. DE 参数设置 ==========================
nVar = length(data.DFenPei);  
popSize = 60;
maxGen = 150;
weights.w1 = 0.001; weights.w2 = 1.0; 
userData.vMax = 1.2; 

% ========================== 3. 执行 DE 优化 ==========================
fprintf('DE 优化启动，正在生成统一规格图表...\n');
pop = rand(popSize, nVar);
fitness = zeros(popSize, 1);
for i = 1:popSize
    fitness(i) = fitness_function(pop(i,:), data, weights);
end

[bestScore, bestIdx] = min(fitness);
zbest = pop(bestIdx, :);
cg_curve = zeros(1, maxGen);

for g = 1:maxGen
    for i = 1:popSize
        A = randperm(popSize, 4); A(A==i) = [];
        v = pop(A(1),:) + 0.5 * (pop(A(2),:) - pop(A(3),:));
        j0 = randi(nVar);
        for j = 1:nVar
            if rand < 0.3 || j == j0
                target_pos = v(j);
                if target_pos > 1, target_pos = 1; end
                if target_pos < 0, target_pos = 0; end
                pop(i,j) = target_pos;
            end
        end
        f_new = fitness_function(pop(i,:), data, weights);
        if f_new < fitness(i)
            fitness(i) = f_new;
            if f_new < bestScore
                bestScore = f_new;
                zbest = pop(i,:);
            end
        end
    end
    cg_curve(g) = bestScore;
end

% ========================== 4. 结果可视化 (统一学术格式) ==========================
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P); 

% --- 图 1: 2D 空间分配地图 (高占比填充版) ---
figure('Color','w', 'Name', 'DE 2D Allocation', 'Position', [100, 100, 900, 800]); 
hold on; box on;

% 1. 绘制道路底图
if isfield(data, 'road')
    for i = 1:length(data.road)
        plot(data.road{i}(:,1)-offset_x, data.road{i}(:,2)-offset_y, ...
             'Color', [0.85 0.85 0.85], 'LineWidth', 0.5); 
    end
end

% 2. 绘制所有连线 (全量绘制，统一红色透明)
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1); eIdx = data.DFenPei{i}(X_final(i)+1);
    line([house_x(hIdx), binan_x(eIdx)], [house_y(hIdx), binan_y(eIdx)], ...
         'Color', [1 0 0 0.1], 'LineWidth', 0.3);
end
for i = 1:size(data.FID, 1)
    line([house_x(data.FID(i,1)), binan_x(data.FID(i,2))], ...
         [house_y(data.FID(i,1)), binan_y(data.FID(i,2))], ...
         'Color', [1 0 0 0.1], 'LineWidth', 0.3);
end

% 3. 绘制节点
h_res = scatter(house_x, house_y, 10, [0.0, 0.2, 0.6], 'filled'); 
h_shl = scatter(binan_x, binan_y, 75, 'g', '^', 'filled', 'MarkerEdgeColor', 'k'); 

% 4. 格式化设置
axis equal; axis tight; grid on;
ax = gca; ax.LooseInset = ax.TightInset;
ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
ax.FontSize = 11; ax.FontWeight = 'bold';
title('DE Optimized Shelter Allocation Map', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X Coordinate Offset (m)', 'FontWeight', 'bold');
ylabel('Y Coordinate Offset (m)', 'FontWeight', 'bold');
legend([h_shl, h_res], {'Optimized Shelter', 'Residential Area'}, 'Location', 'northeast');

% --- 图 2: 收敛曲线 ---
figure('Color','w', 'Name', 'DE Convergence', 'Position', [200, 200, 600, 450]);
plot(cg_curve, 'LineWidth', 2.5, 'Color', [0.85, 0.33, 0.1]); 
grid on; box on;
xlabel('Iteration', 'FontWeight', 'bold');
ylabel('Best Fitness (Cost)', 'FontWeight', 'bold');
title('DE Optimization Convergence Curve', 'FontSize', 12);

% --- 图 3: 3D 需求压力地形图 ---
figure('Color','w', 'Name', 'DE 3D Landscape', 'Position', [150, 150, 900, 750]);
hold on; grid on;

% 1. 生成 KDE 地形
res = 80; [Xq, Yq] = meshgrid(linspace(min(house_x), max(house_x), res), linspace(min(house_y), max(house_y), res));
Z = zeros(size(Xq)); bw = (max(house_x) - min(house_x)) / 25;
sample_idx = randperm(length(house_x), min(1200, length(house_x)));
for i = sample_idx
    d2 = (Xq - house_x(i)).^2 + (Yq - house_y(i)).^2;
    Z = Z + exp(-d2 / (2 * bw^2)); 
end

% 2. 绘制地形与色条
surf(Xq, Yq, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.6); 
colormap(jet); shading interp; cb = colorbar; 
ylabel(cb, 'Demand Intensity', 'FontWeight', 'bold');

% 3. 绘制空中避难所投影
used_bin_idx = unique([X_final, data.FID(:,2)']); 
z_top = max(Z(:)) * 1.2; 
for k = used_bin_idx
    plot3([binan_x(k), binan_x(k)], [binan_y(k), binan_y(k)], [0, z_top], ...
          'Color', [1, 0, 0, 0.4], 'LineStyle', '--', 'LineWidth', 0.8);
end
h_3d = scatter3(binan_x(used_bin_idx), binan_y(used_bin_idx), ones(length(used_bin_idx),1)*z_top, 85, ...
                 'g', '^', 'filled', 'MarkerEdgeColor', 'k');

% 4. 3D 修饰
view(-35, 45); 
ax = gca; ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
xlabel('X Coordinate Offset (m)', 'FontWeight', 'bold');
ylabel('Y Coordinate Offset (m)', 'FontWeight', 'bold');
zlabel('Intensity', 'FontWeight', 'bold');
title('DE 3D Demand Landscape & Shelter Location', 'FontSize', 12);
legend(h_3d, 'Optimized Shelter (Air)', 'Location', 'northeast');

fprintf('DE 任务完成：三图已按统一规格生成。\n'); toc;

% ========================== 适应度函数 ==========================
function score = fitness_function(x, S, w)
    x(x < 0.01) = 0.01;
    X = ceil(x .* S.P); 
    total_dist = S.alldis;
    Y = S.YFenPei; 
    for i = 1:length(X)
        hID = S.DFenPei{i}(1); eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12; 
    end
    score = w.w1 * total_dist + w.w2 * var(Y); 
end