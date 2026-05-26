%% GWO 疏散分配优化 - 统一规格增强版
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

% ========================== 2. GWO 参数设置 ==========================
SearchAgents_no = 50; 
Max_iter = 150; 
dim = length(data.DFenPei);
lb = 0; ub = 1;
weights.w1 = 0.001; weights.w2 = 1.0; 

% 初始化 Alpha, Beta, Delta 狼
Alpha_pos = zeros(1, dim); Alpha_score = inf;
Beta_pos = zeros(1, dim); Beta_score = inf;
Delta_pos = zeros(1, dim); Delta_score = inf;

Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
Convergence_curve = zeros(1, Max_iter);

% ========================== 3. 执行 GWO 优化 ==========================
fprintf('GWO 优化启动，正在生成统一规格图表...\n');

for l = 1:Max_iter
    for i = 1:size(Positions, 1)
        % 边界检查
        Positions(i,:) = max(min(Positions(i,:), ub), lb);
        % 计算适应度
        fitness = gwo_obj(Positions(i,:), data, weights);
        
        % 更新 Alpha, Beta, Delta
        if fitness < Alpha_score
            Alpha_score = fitness; Alpha_pos = Positions(i,:);
        elseif fitness > Alpha_score && fitness < Beta_score
            Beta_score = fitness; Beta_pos = Positions(i,:);
        elseif fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score
            Delta_score = fitness; Delta_pos = Positions(i,:);
        end
    end
    
    a = 2 - l * (2 / Max_iter); 
    for i = 1:SearchAgents_no
        for j = 1:dim
            r1 = rand(); r2 = rand();
            A1 = 2 * a * r1 - a; C1 = 2 * r2;
            D_alpha = abs(C1 * Alpha_pos(j) - Positions(i,j));
            X1 = Alpha_pos(j) - A1 * D_alpha;
            
            r1 = rand(); r2 = rand();
            A2 = 2 * a * r1 - a; C2 = 2 * r2;
            D_beta = abs(C2 * Beta_pos(j) - Positions(i,j));
            X2 = Beta_pos(j) - A2 * D_beta;
            
            r1 = rand(); r2 = rand();
            A3 = 2 * a * r1 - a; C3 = 2 * r2;
            D_delta = abs(C3 * Delta_pos(j) - Positions(i,j));
            X3 = Delta_pos(j) - A3 * D_delta;
            
            Positions(i,j) = (X1 + X2 + X3) / 3;
        end
    end
    Convergence_curve(l) = Alpha_score;
end

% ========================== 4. 结果可视化 (统一规格) ==========================
zbest = Alpha_pos;
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P);

% --- 图 1: 2D 空间分配地图 (最大化占比版) ---
figure('Color','w', 'Name', 'GWO 2D Allocation', 'Position', [100, 100, 850, 850]); 
hold on; box on;

% 1. 道路底图
if isfield(data, 'road')
    for i = 1:length(data.road)
        plot(data.road{i}(:,1)-offset_x, data.road{i}(:,2)-offset_y, 'Color', [0.9 0.9 0.9], 'LineWidth', 0.5); 
    end
end

% 2. 全量分配连线 (超细红色透明线 LineWidth=0.3)
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1); eIdx = data.DFenPei{i}(X_final(i)+1);
    line([house_x(hIdx), binan_x(eIdx)], [house_y(hIdx), binan_y(eIdx)], 'Color', [1 0 0 0.15], 'LineWidth', 0.3);
end
for i = 1:size(data.FID, 1)
    line([house_x(data.FID(i,1)), binan_x(data.FID(i,2))], [house_y(data.FID(i,1)), binan_y(data.FID(i,2))], 'Color', [1 0 0 0.15], 'LineWidth', 0.3);
end

% 3. 节点
h_res = scatter(house_x, house_y, 8, [0.0, 0.2, 0.6], 'filled', 'MarkerFaceAlpha', 0.6); 
h_shl = scatter(binan_x, binan_y, 70, 'g', '^', 'filled', 'MarkerEdgeColor', 'k'); 

% 4. 占比最大化设置
axis equal; axis tight; grid on;
ax = gca;
ax.LooseInset = ax.TightInset; % 消除默认白边
set(ax, 'Position', [0.08, 0.08, 0.88, 0.88]); % 强制拉伸占比
ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
ax.FontSize = 11; ax.FontWeight = 'bold';

title('GWO Optimized Shelter Allocation Map', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X Coordinate Offset (m)', 'FontWeight', 'bold');
ylabel('Y Coordinate Offset (m)', 'FontWeight', 'bold');
legend([h_shl, h_res], {'Optimized Shelter', 'Residential Area'}, 'Location', 'northeast');

% --- 图 2: 收敛曲线 ---
figure('Color','w', 'Name', 'GWO Convergence', 'Position', [200, 200, 600, 450]);
plot(Convergence_curve, 'LineWidth', 2.5, 'Color', [0.85, 0.33, 0.1]); 
grid on; box on;
xlabel('Iteration', 'FontWeight', 'bold'); ylabel('Best Fitness (Cost)', 'FontWeight', 'bold');
title('GWO Optimization Convergence Process', 'FontSize', 12);

% --- 图 3: 3D 需求压力地形图 ---
figure('Color','w', 'Name', 'GWO 3D Landscape', 'Position', [150, 150, 900, 750]);
hold on; grid on;

res = 80; [Xq, Yq] = meshgrid(linspace(min(house_x), max(house_x), res), linspace(min(house_y), max(house_y), res));
Z = zeros(size(Xq)); bw = (max(house_x) - min(house_x)) / 25;
sample_idx = randperm(length(house_x), min(1200, length(house_x)));
for i = sample_idx
    d2 = (Xq - house_x(i)).^2 + (Yq - house_y(i)).^2;
    Z = Z + exp(-d2 / (2 * bw^2)); 
end

surf(Xq, Yq, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.6); 
colormap(jet); shading interp; cb = colorbar; 
ylabel(cb, 'Demand Intensity', 'FontWeight', 'bold');

used_bin_idx = unique([X_final, data.FID(:,2)']); 
z_top = max(Z(:)) * 1.2; 
for k = used_bin_idx
    plot3([binan_x(k), binan_x(k)], [binan_y(k), binan_y(k)], [0, z_top], 'Color', [1, 0, 0, 0.4], 'LineStyle', '--', 'LineWidth', 0.8);
end
h_3d = scatter3(binan_x(used_bin_idx), binan_y(used_bin_idx), ones(length(used_bin_idx),1)*z_top, 85, 'g', '^', 'filled', 'MarkerEdgeColor', 'k');

view(-35, 45); ax = gca; ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
xlabel('X Coordinate Offset (m)', 'FontWeight', 'bold'); ylabel('Y Coordinate Offset (m)', 'FontWeight', 'bold');
zlabel('Intensity', 'FontWeight', 'bold');
title('GWO 3D Demand Landscape & Shelter Location', 'FontSize', 12);
legend(h_3d, 'Optimized Shelter (Air)', 'Location', 'northeast');

fprintf('GWO 任务完成：三图已按统一规格生成，占比已优化。\n'); toc;

% ========================== 适应度函数 ==========================
function score = gwo_obj(x, S, w)
    x(x < 0.01) = 0.01;
    X = ceil(x .* S.P); 
    total_dist = S.alldis_fixed;
    Y = S.YFenPei_fixed; 
    for i = 1:length(X)
        hID = S.DFenPei{i}(1); eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12; 
    end
    score = w.w1 * total_dist + w.w2 * var(Y); 
end