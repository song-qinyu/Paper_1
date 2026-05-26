%% CS 布谷鸟算法疏散分配优化 - 字段修复版
clear; close all; clc; tic;

% ========================== 0. 自动路径处理 ==========================
cs_path = 'E:\论文格式\NSGA\SA\布谷鸟'; 
if exist(cs_path, 'dir')
    addpath(cs_path);
end

% ========================== 1. 加载数据 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat');
end

% ！！！核心修复：确保 dis 变量被存入 data 结构体 ！！！
if exist('dis','var')
    data.dis = dis; 
else
    error('sj5.mat 中缺失 dis 距离矩阵');
end

% 坐标偏移处理
raw_x = data.start(:,1); raw_y = data.start(:,2);
offset_x = min(raw_x); offset_y = min(raw_y);
house_x = raw_x - offset_x;
house_y = raw_y - offset_y;
binan_x = data.binan(:,1) - offset_x;
binan_y = data.binan(:,2) - offset_y;

% ========================== 2. 参数设置 ==========================
N = 30;                         
Max_iteration = 150;            
data.P = cellfun(@(x) length(x)-1, data.DFenPei);
dim = length(data.DFenPei);     
lb = 0; ub = 1;
weights.w1 = 0.001; weights.w2 = 1.0; 

% 预处理固定分配
data.alldis_fixed = 0;
data.YFenPei_fixed = zeros(1, size(data.binan, 1));
data.FID = []; 
for k = 1:length(B)
    if length(B{k}) == 1
        targetBinan = B{k};
        data.YFenPei_fixed(targetBinan) = data.YFenPei_fixed(targetBinan) + 12; 
        data.FID = [data.FID; k, targetBinan];
        % 这里也统一使用 data.dis
        data.alldis_fixed = data.alldis_fixed + data.dis(k, targetBinan);
    end
end

% 定义目标函数
fobj = @(x) cs_obj_fixed(x, data, weights);

% ========================== 3. 运行 CS 算法 ==========================
fprintf('CS 优化启动，正在生成统一规格图表...\n');
[best_x, fmin, Convergence_curve] = CS(N, Max_iteration, lb, ub, dim, fobj);

% ========================== 4. 结果可视化 ==========================
zbest = best_x;
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P);

% --- 图 1: 2D 空间分配地图 (占比最大化) ---
figure('Color','w', 'Name', 'CS 2D Allocation', 'Position', [100, 100, 850, 850]); 
hold on; box on;

if isfield(data, 'road')
    for i = 1:length(data.road)
        plot(data.road{i}(:,1)-offset_x, data.road{i}(:,2)-offset_y, 'Color', [0.9 0.9 0.9], 'LineWidth', 0.5); 
    end
end

for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1); eIdx = data.DFenPei{i}(X_final(i)+1);
    line([house_x(hIdx), binan_x(eIdx)], [house_y(hIdx), binan_y(eIdx)], 'Color', [1 0 0 0.15], 'LineWidth', 0.3);
end
for i = 1:size(data.FID, 1)
    line([house_x(data.FID(i,1)), binan_x(data.FID(i,2))], [house_y(data.FID(i,1)), binan_y(data.FID(i,2))], 'Color', [1 0 0 0.15], 'LineWidth', 0.3);
end

h_res = scatter(house_x, house_y, 8, [0.0, 0.2, 0.6], 'filled', 'MarkerFaceAlpha', 0.6); 
h_shl = scatter(binan_x, binan_y, 70, 'g', '^', 'filled', 'MarkerEdgeColor', 'k'); 

axis equal; axis tight; grid on;
ax = gca; ax.LooseInset = ax.TightInset; 
set(ax, 'Position', [0.08, 0.08, 0.88, 0.88]); 
ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
ax.FontSize = 11; ax.FontWeight = 'bold';

title('CS Optimized Shelter Allocation Map', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X Coordinate Offset (m)', 'FontWeight', 'bold');
ylabel('Y Coordinate Offset (m)', 'FontWeight', 'bold');
legend([h_shl, h_res], {'Optimized Shelter', 'Residential Area'}, 'Location', 'northeast');

% --- 图 2: 收敛曲线 ---
figure('Color','w', 'Name', 'CS Convergence', 'Position', [200, 200, 600, 450]);
plot(Convergence_curve, 'LineWidth', 2.5, 'Color', [0.85, 0.33, 0.1]); 
grid on; box on;
xlabel('Iteration', 'FontWeight', 'bold'); ylabel('Best Fitness (Cost)', 'FontWeight', 'bold');
title('CS Optimization Convergence Process', 'FontSize', 12);

% --- 图 3: 3D 需求压力地形图 ---
figure('Color','w', 'Name', 'CS 3D Landscape', 'Position', [150, 150, 900, 750]);
hold on; grid on;
res = 80; [Xq, Yq] = meshgrid(linspace(min(house_x), max(house_x), res), linspace(min(house_y), max(house_y), res));
Z = zeros(size(Xq)); bw = (max(house_x) - min(house_x)) / 25;
sample_idx = randperm(length(house_x), min(1200, length(house_x)));
for i = sample_idx
    d2 = (Xq - house_x(i)).^2 + (Yq - house_y(i)).^2;
    Z = Z + exp(-d2 / (2 * bw^2)); 
end
surf(Xq, Yq, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.6); 
colormap(jet); shading interp; colorbar;
used_bin_idx = unique([X_final, data.FID(:,2)']); 
z_top = max(Z(:)) * 1.2; 
for k = used_bin_idx
    plot3([binan_x(k), binan_x(k)], [binan_y(k), binan_y(k)], [0, z_top], 'Color', [1, 0, 0, 0.4], 'LineStyle', '--', 'LineWidth', 0.8);
end
scatter3(binan_x(used_bin_idx), binan_y(used_bin_idx), ones(length(used_bin_idx),1)*z_top, 85, 'g', '^', 'filled', 'MarkerEdgeColor', 'k');
view(-35, 45); ax = gca; ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Intensity');
title('CS 3D Demand Landscape & Shelter Location', 'FontSize', 12);

fprintf('CS 任务完成：字段报错已修复，三图已生成。\n'); toc;

% ========================== 修复后的适应度函数 ==========================
function score = cs_obj_fixed(x, S, w)
    x(x < 0.01) = 0.01;
    X = ceil(x .* S.P); 
    total_dist = S.alldis_fixed;
    Y = S.YFenPei_fixed; 
    % 确保 S 包含 dis 字段
    for i = 1:length(X)
        hID = S.DFenPei{i}(1); 
        eID = S.DFenPei{i}(X(i)+1);
        % 这里之前报错是因为 S.dis 没被赋值
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12; 
    end
    score = w.w1 * total_dist + w.w2 * var(Y); 
end