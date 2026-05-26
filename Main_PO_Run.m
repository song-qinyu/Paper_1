%% Political Optimizer (PO) 疏散分配优化 - 终极深度收敛版
clc; clear; close all; tic;

% ========================== 1. 加载与变量兼容处理 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat');
end

% 自动识别距离矩阵变量名
if exist('dis','var')
    dist_matrix = dis; 
elseif isfield(data, 'dis')
    dist_matrix = data.dis;
else
    error('数据中未找到距离矩阵变量 (dis 或 data.dis)');
end

% 坐标偏移处理
offset_x = min(data.start(:,1)); offset_y = min(data.start(:,2));
house_x = data.start(:,1) - offset_x;
house_y = data.start(:,2) - offset_y;
binan_x = data.binan(:,1) - offset_x;
binan_y = data.binan(:,2) - offset_y;

dim = length(DFenPei); 

% 识别固定分配点 (用于 2D 全量地图显示)
fixed_data = []; 
if exist('B','var')
    for k = 1:length(B)
        if length(B{k}) == 1
            fixed_data = [fixed_data; k, B{k}]; 
        end
    end
end

% ========================== 2. 终极参数设置 (针对高维收敛) ==========================
parties = 20;                   % 增加政党数：从10增加到20，极大地增强后期收敛能力
areas = parties;               
populationSize = parties * areas; % 种群规模达到 400

% 【评价次数设定】：50万次评价
fEvals = 500000;                
Max_iteration = round(fEvals / (parties * areas + areas)); 
% 此时 Max_iteration 约为 1190 次，但每一代的“含金量”是之前的 4 倍
lambda = 1.0;                  

lb = ones(1, dim); 
ub = zeros(1, dim);
for i = 1:dim
    ub(i) = length(DFenPei{i}) - 1; 
end

% 目标函数句柄
fobj = @(x) fobj_evacuation(x, DFenPei, dist_matrix);

% ========================== 3. 执行 PO 优化 ==========================
fprintf('PO 终极优化启动...\n');
fprintf('评价次数: %d | 政党数量: %d | 预计迭代: %d\n', fEvals, parties, Max_iteration);

[Best_score, Best_pos, Convergence_curve] = PO(populationSize, areas, parties, lambda, Max_iteration, lb, ub, dim, fobj);

X_final = round(Best_pos);
toc;

% ========================== 4. 可视化渲染 ==========================

% --- 图 1: 深度收敛曲线 ---
figure('Name','PO 深度收敛过程','Color','w');
plot(Convergence_curve, 'Color', [0.85 0.32 0.1], 'LineWidth', 2.5);
grid on; xlabel('Iteration'); ylabel('Total Distance (m)');
title(['PO Deep Convergence (Final: ', num2str(Best_score, '%.2f'), ' m)'], 'FontSize', 12);

% --- 图 2: 2D 全量分配地图 (GA 模板风格) ---
figure('Color','w', 'Name', 'PO Final Map', 'Position', [100, 100, 900, 800]); 
hold on; box on;

% 绘制道路底图
if isfield(data, 'road')
    for i = 1:length(data.road)
        plot(data.road{i}(:,1)-offset_x, data.road{i}(:,2)-offset_y, ...
             'Color', [0.85 0.85 0.85], 'LineWidth', 0.5); 
    end
end

% 绘制全量红色分配线
% 1. 优化点 (460个)
for i = 1:dim
    hIdx = DFenPei{i}(1); 
    val = round(X_final(i));
    opt_idx = max(1, min(val, length(DFenPei{i})-1));
    eIdx = DFenPei{i}(opt_idx + 1);
    line([house_x(hIdx), binan_x(eIdx)], [house_y(hIdx), binan_y(eIdx)], ...
         'Color', [1 0 0 0.12], 'LineWidth', 0.35);
end
% 2. 固定点 (2214个)
if ~isempty(fixed_data)
    for i = 1:size(fixed_data, 1)
        line([house_x(fixed_data(i,1)), binan_x(fixed_data(i,2))], ...
             [house_y(fixed_data(i,1)), binan_y(fixed_data(i,2))], ...
             'Color', [1 0 0 0.12], 'LineWidth', 0.35);
    end
end

% 绘制节点
h_res = scatter(house_x, house_y, 10, [0.0, 0.2, 0.6], 'filled'); 
h_shl = scatter(binan_x, binan_y, 75, 'g', '^', 'filled', 'MarkerEdgeColor', 'k'); 

axis equal; axis tight; grid on;
ax = gca; ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
title('PO Optimized Shelter Allocation Map (Full View)', 'FontSize', 13, 'FontWeight', 'bold');
legend([h_shl, h_res], {'Shelter', 'Residential'}, 'Location', 'northeast');

% ========================== 5. 健壮性目标函数 ==========================
function fitness = fobj_evacuation(X, DFenPei, dist_matrix)
    X_idx = round(X);
    fitness = 0;
    for i = 1:length(X_idx)
        opts_len = length(DFenPei{i});
        % 严格边界限位
        idx = X_idx(i);
        if idx < 1, idx = 1; end
        if idx > opts_len - 1, idx = opts_len - 1; end
        
        h_id = DFenPei{i}(1);     
        b_id = DFenPei{i}(idx+1); 
        fitness = fitness + dist_matrix(h_id, b_id);
    end
end