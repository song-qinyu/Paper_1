%% NSGA-II 多目标疏散分配优化 - 统一可视化版
clc; clear; close all; tic;

% ========================== 1. 加载与预处理 ==========================
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

% 初始化固定分配
data.alldis_fixed = 0;
data.YFenPei_fixed = zeros(1, size(data.binan, 1));
data.FID = []; 
for k = 1:length(B)
    if length(B{k}) == 1
        targetBinan = B{k};
        data.YFenPei_fixed(targetBinan) = data.YFenPei_fixed(targetBinan) + 12; 
        data.FID = [data.FID; k, targetBinan];
        data.alldis_fixed = data.alldis_fixed + data.dis(k, targetBinan);
    end
end

% ========================== 2. NSGA-II 参数设置 ==========================
nVar = length(data.DFenPei);  
popSize = 100;   
maxGen  = 100;   

userData.S = data;
options = optimoptions('gamultiobj', ...
    'PopulationSize', popSize, ...
    'MaxGenerations', maxGen, ...
    'Display', 'iter', ...
    'PlotFcn', []); % 禁用默认绘图以使用统一格式

% ========================== 3. 执行 NSGA-II 优化 ==========================
fprintf('NSGA-II 优化启动，正在搜索帕累托最优解...\n');
problemFitness = @(x) nsga_obj_wrapper(x, userData);
lb = zeros(1, nVar); ub = ones(1, nVar);

[xSolutions, fSolutions] = gamultiobj(problemFitness, nVar, [],[],[],[], lb, ub, [], options);

% 从帕累托前沿中自动选择一个解进行绘图（这里选 F1:总距离最短 的方案）
[~, bestIdx] = min(fSolutions(:, 1)); 
zbest = xSolutions(bestIdx, :);
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P);

% ========================== 4. 结果可视化 (统一学术格式) ==========================

% --- 图 1: 2D 空间分配地图 (高占比填充版) ---
figure('Color','w', 'Name', 'NSGA-II 2D Allocation', 'Position', [100, 100, 900, 800]); 
hold on; box on;

% 1. 道路底图
if isfield(data, 'road')
    for i = 1:length(data.road)
        plot(data.road{i}(:,1)-offset_x, data.road{i}(:,2)-offset_y, ...
             'Color', [0.85 0.85 0.85], 'LineWidth', 0.5); 
    end
end

% 2. 全量分配连线 (统一红色)
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

% 3. 节点
h_res = scatter(house_x, house_y, 10, [0.0, 0.2, 0.6], 'filled'); 
h_shl = scatter(binan_x, binan_y, 75, 'g', '^', 'filled', 'MarkerEdgeColor', 'k'); 

% 4. 格式化
axis equal; axis tight; grid on;
ax = gca; ax.LooseInset = ax.TightInset;
ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
ax.FontSize = 11; ax.FontWeight = 'bold';
title('NSGA-II Optimized Allocation Map (Best Distance)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X Coordinate Offset (m)', 'FontWeight', 'bold');
ylabel('Y Coordinate Offset (m)', 'FontWeight', 'bold');
legend([h_shl, h_res], {'Optimized Shelter', 'Residential Area'}, 'Location', 'northeast');

% --- 图 2: Pareto Front (多目标特有图) ---
figure('Color','w', 'Name', 'NSGA-II Pareto Front', 'Position', [200, 200, 600, 450]);
plot(fSolutions(:,1), fSolutions(:,2), 'o', 'MarkerEdgeColor', [0.85, 0.33, 0.1], ...
     'MarkerFaceColor', [1, 0.9, 0.8], 'LineWidth', 1.5);
grid on; box on;
xlabel('F1: Total Evacuation Distance', 'FontWeight', 'bold');
ylabel('F2: Capacity Variance (Balance)', 'FontWeight', 'bold');
title('NSGA-II Pareto Optimal Front', 'FontSize', 12);

% --- 图 3: 3D 需求压力地形图 ---
figure('Color','w', 'Name', 'NSGA-II 3D Landscape', 'Position', [150, 150, 900, 750]);
hold on; grid on;

% 1. 生成 KDE 地形
res = 80; [Xq, Yq] = meshgrid(linspace(min(house_x), max(house_x), res), linspace(min(house_y), max(house_y), res));
Z = zeros(size(Xq)); bw = (max(house_x) - min(house_x)) / 25;
sample_idx = randperm(length(house_x), min(1200, length(house_x)));
for i = sample_idx
    d2 = (Xq - house_x(i)).^2 + (Yq - house_y(i)).^2;
    Z = Z + exp(-d2 / (2 * bw^2)); 
end

% 2. 绘制地形
surf(Xq, Yq, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.6); 
colormap(jet); shading interp; cb = colorbar; 
ylabel(cb, 'Demand Intensity', 'FontWeight', 'bold');

% 3. 绘制空中投影
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
title('NSGA-II 3D Demand Landscape (Selected Solution)', 'FontSize', 12);
legend(h_3d, 'Optimized Shelter (Air)', 'Location', 'northeast');

fprintf('NSGA-II 任务完成：三图已按统一规格生成。\n'); toc;

% ========================== 辅助函数 ==========================
function f = nsga_obj_wrapper(x, userData)
    S = userData.S;
    x(x < 0.01) = 0.01;
    X = ceil(x .* S.P);
    total_dist = S.alldis_fixed;
    Y = S.YFenPei_fixed;
    for i = 1:length(X)
        hID = S.DFenPei{i}(1); eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12; 
    end
    f = [total_dist, var(Y)];
end