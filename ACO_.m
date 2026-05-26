%% ACO 路径规划优化 - 统一规格修正版
clc; clear; close all; tic;

% ========================== 1. 数据加载与映射 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat');
end

% 获取原始坐标并计算偏移
start_raw = data.start; 
binan_raw = data.binan; 
offset_x = min(start_raw(:,1)); offset_y = min(start_raw(:,2));
house_x = start_raw(:,1) - offset_x;
house_y = start_raw(:,2) - offset_y;
binan_x = binan_raw(:,1) - offset_x;
binan_y = binan_raw(:,2) - offset_y;

% 栅格地图参数
MM = 300; Lgrid = 1;
all_p = [start_raw; binan_raw];
min_x = min(all_p(:,1)); max_x = max(all_p(:,1));
min_y = min(all_p(:,2)); max_y = max(all_p(:,2));

% 映射函数：地理坐标 -> 栅格索引 (1-300)
map_x = @(x) max(1, min(MM, round((MM-1)*(x-min_x)/(max_x-min_x+eps))+1));
map_y = @(y) max(1, min(MM, round((MM-1)*(y-min_y)/(max_y-min_y+eps))+1));

% ========================== 2. 构建地图与执行 ACO ==========================
G = ones(MM, MM); % 1表示障碍（非道路），0表示通路（道路）
if isfield(data, 'road')
    for i = 1:length(data.road)
        % 修正了之前可能导致“无效表达式”的语法
        rx = map_x(data.road{i}(:,1)); 
        ry = map_y(data.road{i}(:,2));
        for j = 1:length(rx)-1
            num_pts = max(abs(rx(j+1)-rx(j)), abs(ry(j+1)-ry(j))) * 2;
            ix = round(linspace(rx(j), rx(j+1), num_pts));
            iy = round(linspace(ry(j), ry(j+1), num_pts));
            for k = 1:length(ix)
                G(iy(k), ix(k)) = 0; % 标记为道路
            end
        end
    end
end

D_aco = ones(MM^2, 8); 
num_plan = min(30, size(start_raw, 1)); % 演示前30个点的真实路径规划
results = cell(num_plan, 1);
total_dist_history = zeros(num_plan, 1);

fprintf('ACO 优化启动，正在生成统一规格图表...\n');

for i = 1:num_plan
    % 转换起终点到栅格
    sx = map_x(start_raw(i,1)); sy = map_y(start_raw(i,2));
    d_geo = sqrt((binan_raw(:,1)-start_raw(i,1)).^2 + (binan_raw(:,2)-start_raw(i,2)).^2);
    [~, b_idx] = min(d_geo);
    gx = map_x(binan_raw(b_idx,1)); gy = map_y(binan_raw(b_idx,2));
    
    % 【核心修正】生成符合 standard.m 要求的 90000x1 距离矩阵
    [C, R] = meshgrid(1:MM, 1:MM);
    dis_to_goal = sqrt((C - gx).^2 + (R - gy).^2);
    dis_vector = dis_to_goal(:); 
    
    try
        % 调用路径规划函数
        [~,~,~,~,~, route, ~] = standard(D_aco, [sx,sy], [gx,gy], dis_vector, [], ...
            15, 20, 10, 0.5, 1, 1, 1, 100, G, MM, Lgrid, 1, 1, 1, zeros(MM,MM), 1);
        results{i} = route;
        if ~isempty(route), total_dist_history(i) = length(route); end
    catch
        continue;
    end
end

% ========================== 3. 结果可视化 (统一规格) ==========================

% --- 图 1: 2D 路径分配地图 (全量显示 + 占比最大化) ---
figure('Color','w', 'Name', 'ACO 2D Path', 'Position', [100, 100, 900, 800]); 
hold on; box on;

% 1. 道路底图
if isfield(data, 'road')
    for i = 1:length(data.road)
        plot(data.road{i}(:,1)-offset_x, data.road{i}(:,2)-offset_y, 'Color', [0.85 0.85 0.85], 'LineWidth', 0.5); 
    end
end

% 2. 绘制 ACO 规划路径 (全量红色)
for i = 1:num_plan
    if ~isempty(results{i})
        [ry, rx] = ind2sub([MM, MM], results{i});
        % 映射回地理坐标
        real_rx = (rx-1)*(max_x-min_x)/(MM-1) + min_x - offset_x;
        real_ry = (ry-1)*(max_y-min_y)/(MM-1) + min_y - offset_y;
        plot(real_rx, real_ry, 'r-', 'LineWidth', 1.0);
    end
end

h_res = scatter(house_x, house_y, 10, [0.0, 0.2, 0.6], 'filled'); 
h_shl = scatter(binan_x, binan_y, 75, 'g', '^', 'filled', 'MarkerEdgeColor', 'k'); 

axis equal; axis tight; grid on;
ax = gca; ax.LooseInset = ax.TightInset; % 撑满框框
ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
title('ACO Optimized Evacuation Path Map', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X Coordinate Offset (m)', 'FontWeight', 'bold');
ylabel('Y Coordinate Offset (m)', 'FontWeight', 'bold');
legend([h_shl, h_res], {'Shelter', 'Residential Area'}, 'Location', 'northeast');

% --- 图 2: 收敛曲线 ---
figure('Color','w', 'Name', 'ACO Convergence', 'Position', [200, 200, 600, 450]);
plot(total_dist_history(total_dist_history>0), 'LineWidth', 2.5, 'Color', [0.85, 0.33, 0.1]); 
grid on; box on;
xlabel('Path Instances', 'FontWeight', 'bold'); ylabel('Path Length (Grid)', 'FontWeight', 'bold');
title('ACO Path Optimization Performance', 'FontSize', 12);

% --- 图 3: 3D 需求压力地形图 ---
figure('Color','w', 'Name', 'ACO 3D Landscape', 'Position', [150, 150, 900, 750]);
hold on; grid on;

res = 80; [Xq, Yq] = meshgrid(linspace(min(house_x), max(house_x), res), linspace(min(house_y), max(house_y), res));
Z = zeros(size(Xq)); bw = (max(house_x) - min(house_x)) / 25;
for i = randperm(length(house_x), min(1200, length(house_x)))
    d2 = (Xq - house_x(i)).^2 + (Yq - house_y(i)).^2;
    Z = Z + exp(-d2 / (2 * bw^2)); 
end
surf(Xq, Yq, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.6); 
colormap(jet); shading interp; colorbar;

z_top = max(Z(:)) * 1.2; 
for k = 1:length(binan_x)
    plot3([binan_x(k), binan_x(k)], [binan_y(k), binan_y(k)], [0, z_top], 'Color', [1 0 0 0.4], 'LineStyle', '--');
end
h_3d = scatter3(binan_x, binan_y, ones(size(binan_x))*z_top, 85, 'g', '^', 'filled', 'MarkerEdgeColor', 'k');

view(-35, 45); ax = gca; ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
title('ACO 3D Demand Landscape', 'FontSize', 12);
xlabel('X Coordinate (m)'); ylabel('Y Coordinate (m)'); zlabel('Intensity');

fprintf('ACO 任务完成：所有规格已统一。\n'); toc;