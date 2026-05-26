clc; clear; close all; tic;

%% 1. 数据加载与对齐
load('sj5.mat'); 
num_centers = 57; 
CityNum = size(dis, 2); 

% 统一坐标偏移
off_x = min(data.start(:,1)); off_y = min(data.start(:,2));
hx = data.start(:,1) - off_x; hy = data.start(:,2) - off_y;
all_sx = [s.X] - off_x; all_sy = [s.Y] - off_y;

Capacity = [s.Capacity]';             
Demand = ones(size(dis, 1), 1); 

%% 2. 暴力 SA 参数 (保持动感收敛的核心)
T0 = 1e6;       % 提高初始温度
Tend = 1;    
L = 80;         
q = 0.92;       % 快速降温但在内循环增加扰动

% 初始解
S1 = randperm(CityNum, num_centers); 
best_S = S1; 
[best_dist, ~] = Evaluation_BC_Fixed(S1, dis, Demand, Capacity);
trace = []; 
no_improve_count = 0;

%% 3. 模拟退火循环
fprintf('SA 优化强制启动 (收敛曲线动态优化中)...\n');
count = 0;
while T0 > Tend
    count = count + 1;
    improved_in_L = false;
    
    for i = 1:L
        % 生成新解
        S2 = S1; unselected = setdiff(1:CityNum, S1);
        
        % --- 关键：根据陷入时间增加扰动强度 ---
        num_swap = 1;
        if no_improve_count > 15, num_swap = 5; end % 连续不改进则加大力度
        
        for k = 1:num_swap
            swap_idx = randi(num_centers);
            S2(swap_idx) = unselected(randi(length(unselected)));
            unselected = setdiff(1:CityNum, S2);
        end
        
        [d2, ~] = Evaluation_BC_Fixed(S2, dis, Demand, Capacity);
        
        % Metropolis 准则
        delta = d2 - best_dist;
        if d2 < best_dist || exp(-delta / T0) > rand
            S1 = S2;
            if d2 < best_dist
                best_dist = d2;
                best_S = S2;
                improved_in_L = true;
            end
        end
    end
    
    if ~improved_in_L, no_improve_count = no_improve_count + 1; else, no_improve_count = 0; end
    
    trace(count) = best_dist; 
    T0 = T0 * q;
    if mod(count, 10) == 0, fprintf('周期: %d, 当前最佳距离: %.2e\n', count, best_dist); end
end

%% 4. 三图标准输出 (强化坐标轴与色条)
[~, best_assign] = Evaluation_BC_Fixed(best_S, dis, Demand, Capacity);
best_centers_x = all_sx(best_S); best_centers_y = all_sy(best_S);

% --- 图 1: 2D 分配 (含标注) ---
figure('Color','w','Name','SA 2D'); hold on; box on;
if isfield(data, 'road')
    for k = 1:length(data.road), plot(data.road{k}(:,1)-off_x, data.road{k}(:,2)-off_y, 'Color', [0.85 0.85 0.85]); end
end
for i = 1:length(hx)
    target_idx = best_S(best_assign(i));
    line([hx(i), all_sx(target_idx)], [hy(i), all_sy(target_idx)], 'Color', [1 0 0 0.15]);
end
h_res = scatter(hx, hy, 12, [0.0, 0.2, 0.6], 'filled'); 
h_shl = scatter(best_centers_x, best_centers_y, 70, 'g^', 'filled', 'MarkerEdgeColor', 'k');

% 【视觉强化点 1】：XY轴标注与禁用科学计数法
xlabel('X Coordinate Offset (m)', 'FontWeight', 'bold');
ylabel('Y Coordinate Offset (m)', 'FontWeight', 'bold');
ax = gca; ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0; 

axis equal; grid on; title('SA Algorithm: Shelter Allocation (2D)');
legend([h_shl, h_res], {'Optimized Shelter', 'Residential Area'}, 'Location', 'northeast');

% --- 图 2: 收敛曲线 ---
figure('Color','w','Name','SA Convergence');
plot(trace, 'LineWidth', 2, 'Color', [0.85, 0.33, 0.1]);
grid on; xlabel('Iteration'); ylabel('Fitness (Total Distance)'); 
title('SA Convergence Curve (Breakthrough)');

% --- 图 3: 3D 地形 (不含宅基地点) ---
figure('Color','w','Name','SA 3D View'); hold on; grid on;
res = 80; [Xq, Yq] = meshgrid(linspace(min(hx), max(hx), res), linspace(min(hy), max(hy), res));
Z = zeros(size(Xq)); bw = (max(hx)-min(hx))/25;
for i = randperm(length(hx), min(1000, length(hx))), Z = Z + exp(-((Xq-hx(i)).^2 + (Yq-hy(i)).^2) / (2*bw^2)); end

surf(Xq, Yq, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.6); 
colormap(jet); shading interp; 

% 【视觉强化点 2】：添加右侧色条并标注
cb = colorbar; 
ylabel(cb, 'Demand Intensity (Population Density)', 'FontSize', 10); 

z_top = max(Z(:)) * 1.2;
for k = 1:num_centers, plot3([best_centers_x(k), best_centers_x(k)], [best_centers_y(k), best_centers_y(k)], [0, z_top], 'r--', 'LineWidth', 0.8); end
h_shl3d = scatter3(best_centers_x, best_centers_y, ones(1,num_centers)*z_top, 80, 'g^', 'filled', 'MarkerEdgeColor', 'k');

% 【视觉强化点 3】：3D坐标轴标注
xlabel('X Offset (m)'); ylabel('Y Offset (m)'); zlabel('Intensity');
ax3 = gca; ax3.XAxis.Exponent = 0; ax3.YAxis.Exponent = 0; 

view(-35, 45); axis tight; title('SA 3D Landscape (Shelters Only)');
legend(h_shl3d, 'Optimized Shelter (Air)', 'Location', 'northeast');
toc;

function [total_d, assign] = Evaluation_BC_Fixed(S, dis_matrix, Demand, Cap)
    sub_dis = dis_matrix(:, S); [val, local_pos] = sort(sub_dis, 2);
    current_cap = zeros(size(dis_matrix, 2), 1); assign = zeros(size(dis_matrix, 1), 1); total_d = 0;
    for i = 1:size(dis_matrix, 1)
        matched = false;
        for j = 1:length(S)
            idx_in_S = local_pos(i, j); real_idx = S(idx_in_S);
            if current_cap(real_idx) + Demand(i) <= Cap(real_idx)
                current_cap(real_idx) = current_cap(real_idx) + Demand(i);
                total_d = total_d + val(i, j); assign(i) = idx_in_S; matched = true; break;
            end
        end
        if ~matched, total_d = total_d + 1e9; end 
    end
end