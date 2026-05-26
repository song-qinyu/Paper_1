%% PSO 疏散分配优化 - 统一规格修正版
clc; clear; close all; tic;

% ========================== 1. 加载与适配数据 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat');
end

% 坐标偏移处理
raw_x = data.start(:,1); raw_y = data.start(:,2);
offset_x = min(raw_x); offset_y = min(raw_y);
house_x = raw_x - offset_x;
house_y = raw_y - offset_y;
binan_x = data.binan(:,1) - offset_x;
binan_y = data.binan(:,2) - offset_y;

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

% ========================== 2. PSO 参数设置 ==========================
nVar = length(data.DFenPei);  
nPop = 50;           
MaxIt = 150;         
w = 0.8; c1 = 1.5; c2 = 1.5;
weights.w1 = 0.001; weights.w2 = 1.0; 

% ========================== 3. 执行 PSO 优化 ==========================
fprintf('PSO 优化启动，正在生成统一规格图表...\n');
% 修正结构体字段定义
particle = repmat(struct('x',[],'v',[],'best_x',[],'best_cost',[],'cost',[]), nPop, 1);
gBest.cost = inf;

for i = 1:nPop
    particle(i).x = rand(1, nVar);
    particle(i).v = zeros(1, nVar);
    particle(i).cost = pso_obj(particle(i).x, data, weights);
    particle(i).best_x = particle(i).x;
    particle(i).best_cost = particle(i).cost; % 修正点：初始化个体最佳成本
    if particle(i).cost < gBest.cost
        gBest.x = particle(i).x;
        gBest.cost = particle(i).cost;
    end
end

cg_curve = zeros(1, MaxIt);
for it = 1:MaxIt
    for i = 1:nPop
        % 速度更新
        particle(i).v = w*particle(i).v + c1*rand*(particle(i).best_x - particle(i).x) ...
                                       + c2*rand*(gBest.x - particle(i).x);
        % 位置更新
        particle(i).x = particle(i).x + particle(i).v;
        particle(i).x = max(min(particle(i).x, 1), 0);
        
        % 评价
        particle(i).cost = pso_obj(particle(i).x, data, weights);
        
        % 修正点：使用正确的字段名 best_cost 进行比较
        if particle(i).cost < particle(i).best_cost
            particle(i).best_x = particle(i).x;
            particle(i).best_cost = particle(i).cost;
            if particle(i).best_cost < gBest.cost
                gBest.x = particle(i).best_x;
                gBest.cost = particle(i).best_cost;
            end
        end
    end
    cg_curve(it) = gBest.cost;
end

% ========================== 4. 结果可视化 (统一学术格式) ==========================
zbest = gBest.x;
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P);

% --- 图 1: 2D 空间分配地图 (全量显示 + 占比最大化) ---
figure('Color','w', 'Name', 'PSO 2D Allocation', 'Position', [100, 100, 900, 800]); 
hold on; box on;

if isfield(data, 'road')
    for i = 1:length(data.road)
        plot(data.road{i}(:,1)-offset_x, data.road{i}(:,2)-offset_y, 'Color', [0.85 0.85 0.85], 'LineWidth', 0.5); 
    end
end

% 全量连线 (统一红色透明)
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1); eIdx = data.DFenPei{i}(X_final(i)+1);
    line([house_x(hIdx), binan_x(eIdx)], [house_y(hIdx), binan_y(eIdx)], 'Color', [1 0 0 0.1], 'LineWidth', 0.3);
end
for i = 1:size(data.FID, 1)
    line([house_x(data.FID(i,1)), binan_x(data.FID(i,2))], [house_y(data.FID(i,1)), binan_y(data.FID(i,2))], 'Color', [1 0 0 0.1], 'LineWidth', 0.3);
end

h_res = scatter(house_x, house_y, 10, [0.0, 0.2, 0.6], 'filled'); 
h_shl = scatter(binan_x, binan_y, 75, 'g', '^', 'filled', 'MarkerEdgeColor', 'k'); 

axis equal; axis tight; grid on;
ax = gca; ax.LooseInset = ax.TightInset; % 撑满框框
ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
ax.FontSize = 11; ax.FontWeight = 'bold';
title('PSO Optimized Shelter Allocation Map', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X Coordinate Offset (m)', 'FontWeight', 'bold');
ylabel('Y Coordinate Offset (m)', 'FontWeight', 'bold');
legend([h_shl, h_res], {'Optimized Shelter', 'Residential Area'}, 'Location', 'northeast');

% --- 图 2: 收敛曲线 ---
figure('Color','w', 'Name', 'PSO Convergence', 'Position', [200, 200, 600, 450]);
plot(cg_curve, 'LineWidth', 2.5, 'Color', [0.85, 0.33, 0.1]); 
grid on; box on;
xlabel('Iteration', 'FontWeight', 'bold'); ylabel('Best Fitness (Cost)', 'FontWeight', 'bold');
title('PSO Optimization Convergence Process', 'FontSize', 12);

% --- 图 3: 3D 需求压力地形图 ---
figure('Color','w', 'Name', 'PSO 3D Landscape', 'Position', [150, 150, 900, 750]);
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
title('PSO 3D Demand Landscape & Shelter Location', 'FontSize', 12);
legend(h_3d, 'Optimized Shelter (Air)', 'Location', 'northeast');

fprintf('PSO 任务完成：三图已按统一规格生成。\n'); toc;

% ========================== 适应度函数 ==========================
function score = pso_obj(x, S, w)
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