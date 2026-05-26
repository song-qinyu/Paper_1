clc; clear; close all;

%% 1. 数据加载与映射
load('sj5.mat'); 
start_raw = data.start; 
binan_raw = data.binan; 
MM = 300; % 建议先设为100，1000会导致内存溢出卡死
Lgrid = 1;

% 建立地理坐标到栅格的映射
all_p = [start_raw; binan_raw];
min_x = min(all_p(:,1)); max_x = max(all_p(:,1));
min_y = min(all_p(:,2)); max_y = max(all_p(:,2));

map_x = @(x) max(1, min(MM, round((MM-1)*(x-min_x)/(max_x-min_x+eps))+1));
map_y = @(y) max(1, min(MM, round((MM-1)*(y-min_y)/(max_y-min_y+eps))+1));

%% 2. 构建地图 G 与代价矩阵 D
G = ones(MM, MM); % 初始化：1为障碍/非道路

% --- 【核心修改：加载真实道路文件】 ---
try
    % 请确保路径正确
    road = shaperead('E:\论文格式\NSGA\road_2.shp'); 
    fprintf('成功读取道路文件，正在转换地图...\n');
    for i = 1:length(road)
        rx_raw = road(i).X;
        ry_raw = road(i).Y;
        % 过滤NaN并映射
        valid = ~isnan(rx_raw);
        rx_grid = arrayfun(map_x, rx_raw(valid));
        ry_grid = arrayfun(map_y, ry_raw(valid));
        for j = 1:length(rx_grid)
            G(ry_grid(j), rx_grid(j)) = 0; % 设为道路
        end
    end
    % 道路加粗（膨胀），防止断路导致规划失败
    G = double(imdilate(G == 0, strel('disk', 1)) == 0);
catch
    warning('未找到道路文件，将使用模拟路网');
    G(round(MM/2), :) = 0; G(:, round(MM/2)) = 0; 
end

% 确保起终点是通的
for i=1:size(binan_raw,1), G(map_y(binan_raw(i,2)), map_x(binan_raw(i,1))) = 0; end
for i=1:size(start_raw,1), G(map_y(start_raw(i,2)), map_x(start_raw(i,1))) = 0; end

% 构建代价矩阵 D
Dir = [-MM-1, -1, MM-1, MM, MM+1, 1, 1-MM, -MM];
D = inf * ones(MM^2, 8);
fprintf('正在计算代价矩阵，请稍候...\n');
for i = 1:MM^2
    [r, c] = ind2sub([MM, MM], i);
    for d = 1:8
        ni = i + Dir(d);
        if ni > 0 && ni <= MM^2
            [nr, nc] = ind2sub([MM, MM], ni);
            if abs(nr-r)<=1 && abs(nc-c)<=1
                dist = (mod(d,2)==0)*Lgrid + (mod(d,2)~=0)*sqrt(2)*Lgrid;
                % 软约束：在 G=0 (道路) 上走代价极低
                if G(nr, nc) == 0
                    D(i, d) = dist;      
                else
                    D(i, d) = dist * 50; % 非道路代价放大50倍，强制蚂蚁找路
                end
            end
        end
    end
end

%% 3. 循环规划 (规划前 30 个点)
num_plan = 2674; 
results = cell(num_plan, 1);
Uatt = zeros(MM^2, 8); Eta_hobs = zeros(MM^2, 8);

for i = 1:num_plan
    sx = map_x(start_raw(i,1)); sy = map_y(start_raw(i,2));
    % 寻找最近避难所
    d_geo = sqrt((binan_raw(:,1)-start_raw(i,1)).^2 + (binan_raw(:,2)-start_raw(i,2)).^2);
    [~, b_idx] = min(d_geo);
    gx = map_x(binan_raw(b_idx,1)); gy = map_y(binan_raw(b_idx,2));
    
    % 距离启发式矩阵
    dis_matrix = zeros(MM^2, 1);
    for n = 1:MM^2
        [nr, nc] = ind2sub([MM, MM], n);
        dis_matrix(n) = sqrt((nc-gx)^2 + (nr-gy)^2);
    end
    
    fprintf('正在规划宅基地 %d/%d -> 避难所 %d\n', i, num_plan, b_idx);
    try
        % 降低了迭代次数 NC_max=30 和 蚂蚁数 m=20 以提升 30 个点的规划速度
        [~,~,~,~,~, route, ~] = standard(D, [sx,sy], [gx,gy], dis_matrix, [], ...
            30, 20, 10, 0.5, 1, 1, 1, 100, G, MM, Lgrid, 1, 1, 1, Uatt, Eta_hobs);
        results{i} = route;
    catch
        results{i} = [];
    end
end

%% 4. 绘图
figure('Color', 'w', 'Name', '基于SHP道路的蚁群规划');
imagesc([1 MM], [1 MM], G); colormap([1 1 1; 0.9 0.9 0.9]); hold on;
set(gca, 'YDir', 'normal');

% 画避难所 (绿三角)
plot(arrayfun(map_x, binan_raw(:,1)), arrayfun(map_y, binan_raw(:,2)), 'g^', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
% 画宅基地 (红点)
plot(arrayfun(map_x, start_raw(1:num_plan,1)), arrayfun(map_y, start_raw(1:num_plan,2)), 'r.', 'MarkerSize', 12);

% 画路径
for i = 1:num_plan
    if ~isempty(results{i})
        [ry, rx] = ind2sub([MM, MM], results{i});
        % 稍微加一点随机扰动，防止多条路重合看不到
        plot(rx + (rand-0.5)*0.3, ry + (rand-0.5)*0.3, 'LineWidth', 1.5);
    end
end
title(['ACO 沿路规划 (', num2str(num_plan), '个点)']); axis equal; grid on;