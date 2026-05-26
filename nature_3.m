%% Nature-inspired 算法分配对比 - 增量差异可视化版
%  基准图层: Cuckoo Search (CS) - 红色实线
%  差异图层: Firefly Algorithm (FA) - 紫色线条 (仅显示与CS不同的路径)
%  差异图层: Artificial Bee Colony (ABC) - 蓝色线条 (仅显示与CS不同的路径)

clc; clear; close all; tic;

% ======================================================================
%  1. 加载数据与环境准备
% ======================================================================
if ~exist('sj5.mat','file')
    error('未找到数据文件 sj5.mat');
end
load('sj5.mat');

% 统一坐标偏移
raw_x = data.start(:,1); raw_y = data.start(:,2);
offset_x = min(raw_x);   offset_y = min(raw_y);
house_x = raw_x - offset_x; house_y = raw_y - offset_y;
binan_x = data.binan(:,1) - offset_x; binan_y = data.binan(:,2) - offset_y;

% 字段适配
if exist('dis','var'), data.dis = dis; end
data.P = cellfun(@(x) length(x)-1, data.DFenPei);

% ======================================================================
%  2. 获取三种算法的分配方案 (X_final)
%  提示：这里假设您已有运行好的结果，若无，此处应调用各算法函数
% ======================================================================
% [此处模拟三算法运行结果，实际使用时请替换为您的函数输出]
% X_CS  = run_CS(data); 
% X_FA  = run_FA(data);
% X_ABC = run_ABC(data);

% 暂时使用模拟数据进行演示 (实际请替换为您的计算结果变量)
N_house = length(house_x);
% 假设 X_CS, X_FA, X_ABC 已经存在且为 1xN_house 的向量

% ======================================================================
%  3. 核心绘图：叠加对比图
% ======================================================================
figure('Color', 'w', 'Name', 'Nature-inspired Allocation Comparison');
hold on; box on; grid on;

% --- 绘制底层路网/底图 ---
% [此处可添加您原代码中的道路底图绘制逻辑]

% --- A. 绘制 CS 分配线 (基准底图 - 红色) ---
for i = 1:N_house
    target_bin = X_CS(i);
    line([house_x(i), binan_x(target_bin)], [house_y(i), binan_y(target_bin)], ...
         'Color', [1, 0.7, 0.7], 'LineWidth', 0.5, 'LineStyle', '-'); 
end

% --- B. 绘制 FA 差异线 (紫色) ---
diff_fa_count = 0;
for i = 1:N_house
    if X_FA(i) ~= X_CS(i)  % 仅当与 CS 不同时才画线
        target_bin = X_FA(i);
        h_fa = line([house_x(i), binan_x(target_bin)], [house_y(i), binan_y(target_bin)], ...
             'Color', [0.6, 0.2, 0.8], 'LineWidth', 1.2, 'LineStyle', '-');
        diff_fa_count = diff_fa_count + 1;
    end
end

% --- C. 绘制 ABC 差异线 (蓝色) ---
diff_abc_count = 0;
for i = 1:N_house
    if X_ABC(i) ~= X_CS(i) % 仅当与 CS 不同时才画线
        target_bin = X_ABC(i);
        h_abc = line([house_x(i), binan_x(target_bin)], [house_y(i), binan_y(target_bin)], ...
              'Color', [0.2, 0.4, 1.0], 'LineWidth', 1.2, 'LineStyle', '-');
        diff_abc_count = diff_abc_count + 1;
    end
end

% --- 绘制节点 ---
scatter(house_x, house_y, 5, [0.4, 0.4, 0.4], 'filled', 'MarkerFaceAlpha', 0.3);
h_bin = scatter(binan_x, binan_y, 80, 'g', '^', 'filled', 'MarkerEdgeColor', 'k');

% --- 图例与标签 ---
h_cs = plot(NaN, NaN, 'Color', [1, 0.7, 0.7], 'LineWidth', 1.5);
title('Nature-inspired Allocation Comparison: CS (Base) vs FA/ABC (Diff)');
xlabel('X Coordinate Offset (m)'); ylabel('Y Coordinate Offset (m)');

legend([h_cs, h_fa, h_abc, h_bin], ...
    {'CS allocation (Base)', ...
     ['FA allocation (differs from CS, n=', num2str(diff_fa_count), ')'], ...
     ['ABC allocation (differs from CS, n=', num2str(diff_abc_count), ')'], ...
     'Shelter'}, 'Location', 'northeastoutside');

hold off;
fprintf('对比完成。FA 差异点: %d, ABC 差异点: %d\n', diff_fa_count, diff_abc_count);
toc;