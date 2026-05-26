%% 三算法疏散分配对比 - CS / ABC / FA 一键运行版
%  以 CS（布谷鸟算法）为基准，对比 ABC（人工蜂群）和 FA（萤火虫算法）
%  依赖数据文件: sj5.mat（与本脚本放同一目录）
%  依赖外部函数: CS.m（布谷鸟主循环，需放同一目录或已在路径中）
%
%  输出图:
%    图1  CS  收敛曲线
%    图2  ABC 收敛曲线
%    图3  FA  收敛曲线
%    图4  三算法叠加对比地图（核心输出）
%
clc; clear; close all; tic;

% ======================================================================
%  0. 加载数据
% ======================================================================
if ~exist('sj5.mat','file')
    error('未找到数据文件 sj5.mat，请将其与本脚本放在同一目录。');
end
load('sj5.mat');

% 坐标偏移（统一绘图基准）
raw_x    = data.start(:,1);   raw_y  = data.start(:,2);
offset_x = min(raw_x);        offset_y = min(raw_y);
house_x  = raw_x - offset_x;  house_y  = raw_y - offset_y;
binan_x  = data.binan(:,1) - offset_x;
binan_y  = data.binan(:,2) - offset_y;

% 适配字段
if exist('dis','var'), data.dis = dis; end
data.P = cellfun(@(x) length(x)-1, data.DFenPei);

% 固定分配点预处理（三算法共用）
data.alldis  = 0;
data.YFenPei = zeros(1, size(data.binan,1));
data.FID     = [];
for k = 1:length(B)
    if length(B{k}) == 1
        tb = B{k};
        data.YFenPei(tb) = data.YFenPei(tb) + 12;
        data.FID  = [data.FID; k, tb];
        data.alldis = data.alldis + data.dis(k, tb);
    end
end
nVar = length(data.DFenPei);

% 权重（三算法统一）
weights.w1 = 0.001;
weights.w2 = 1.0;

fprintf('数据加载完毕，灵活分配点: %d，固定分配点: %d\n', nVar, size(data.FID,1));

% ======================================================================
%  1. CS 布谷鸟算法（基准算法）
% ======================================================================
fprintf('\n========== [1/3] CS 布谷鸟算法 ==========\n');

cs_N         = 30;
cs_MaxIter   = 1000;
lb = 0; ub = 1;
fobj_cs = @(x) fitness_fn(x, data, weights);

[cs_zbest, cs_bestScore, cs_curve] = CS(cs_N, cs_MaxIter, lb, ub, nVar, fobj_cs);

cs_zbest(cs_zbest < 0.01) = 0.01;
X_CS   = ceil(cs_zbest .* data.P);
eID_CS = zeros(1, nVar);
for i = 1:nVar
    eID_CS(i) = data.DFenPei{i}(X_CS(i)+1);
end
fprintf('CS 完成，最优适应度: %.4f\n', cs_bestScore);

% ======================================================================
%  2. ABC 人工蜂群算法
% ======================================================================
fprintf('\n========== [2/3] ABC 人工蜂群算法 ==========\n');

abc_NP         = 30;
abc_FoodNum    = abc_NP / 2;
abc_limit      = 20;
abc_MaxCycle   = 1000;

% 初始化食物源
abc_Foods  = rand(abc_FoodNum, nVar);
abc_ObjVal = zeros(abc_FoodNum, 1);
for i = 1:abc_FoodNum
    abc_ObjVal(i) = fitness_fn(abc_Foods(i,:), data, weights);
end
abc_Fitness = 1 ./ (1 + abc_ObjVal);
abc_trial   = zeros(1, abc_FoodNum);
abc_curve   = zeros(1, abc_MaxCycle);

[abc_bestScore, abc_best_idx] = min(abc_ObjVal);
abc_zbest = abc_Foods(abc_best_idx, :);

for iter = 1:abc_MaxCycle
    % 雇佣蜂阶段
    for i = 1:abc_FoodNum
        k = floor(rand * abc_FoodNum) + 1;
        while k == i, k = floor(rand * abc_FoodNum) + 1; end
        phi     = rand(1, nVar) * 2 - 1;
        newFood = abc_Foods(i,:) + phi .* (abc_Foods(i,:) - abc_Foods(k,:));
        newFood = min(max(newFood, lb), ub);
        newObj  = fitness_fn(newFood, data, weights);
        if newObj < abc_ObjVal(i)
            abc_Foods(i,:) = newFood; abc_ObjVal(i) = newObj; abc_trial(i) = 0;
        else
            abc_trial(i) = abc_trial(i) + 1;
        end
    end

    % 观察蜂阶段
    abc_Fitness = 1 ./ (1 + abc_ObjVal);
    Prob = abc_Fitness / sum(abc_Fitness);
    for i = 1:abc_FoodNum
        if rand < Prob(i)
            k = floor(rand * abc_FoodNum) + 1;
            while k == i, k = floor(rand * abc_FoodNum) + 1; end
            phi     = rand(1, nVar) * 2 - 1;
            newFood = abc_Foods(i,:) + phi .* (abc_Foods(i,:) - abc_Foods(k,:));
            newFood = min(max(newFood, lb), ub);
            newObj  = fitness_fn(newFood, data, weights);
            if newObj < abc_ObjVal(i)
                abc_Foods(i,:) = newFood; abc_ObjVal(i) = newObj; abc_trial(i) = 0;
            else
                abc_trial(i) = abc_trial(i) + 1;
            end
        end
    end

    % 侦查蜂阶段
    [maxTrial, scout_idx] = max(abc_trial);
    if maxTrial > abc_limit
        abc_Foods(scout_idx,:) = rand(1, nVar);
        abc_ObjVal(scout_idx)  = fitness_fn(abc_Foods(scout_idx,:), data, weights);
        abc_trial(scout_idx)   = 0;
    end

    % 记录最优
    [cur_min, cur_idx] = min(abc_ObjVal);
    if cur_min < abc_bestScore
        abc_bestScore = cur_min;
        abc_zbest     = abc_Foods(cur_idx, :);
    end
    abc_curve(iter) = abc_bestScore;

    if mod(iter, 50) == 0
        fprintf('  ABC 第 %d 代，最优适应度: %.4f\n', iter, abc_bestScore);
    end
end

abc_zbest(abc_zbest < 0.01) = 0.01;
X_ABC   = ceil(abc_zbest .* data.P);
eID_ABC = zeros(1, nVar);
for i = 1:nVar
    eID_ABC(i) = data.DFenPei{i}(X_ABC(i)+1);
end
fprintf('ABC 完成，最优适应度: %.4f\n', abc_bestScore);

% ======================================================================
%  3. FA 萤火虫算法
% ======================================================================
fprintf('\n========== [3/3] FA 萤火虫算法 ==========\n');

fa_n          = 30;
fa_MaxGen     = 1000;
fa_alpha      = 0.5;
fa_betamin    = 0.2;
fa_gamma      = 1.0;

% 初始化萤火虫
fa_zn    = rand(fa_n, nVar);
fa_light = zeros(fa_n, 1);
for i = 1:fa_n
    fa_light(i) = fitness_fn(fa_zn(i,:), data, weights);
end
fa_curve = zeros(1, fa_MaxGen);
[fa_bestScore, fa_best_idx] = min(fa_light);
fa_zbest = fa_zn(fa_best_idx, :);

for k = 1:fa_MaxGen
    % 按亮度升序排列（亮度越低=适应度越好）
    [fa_light, fa_idx] = sort(fa_light);
    fa_zn = fa_zn(fa_idx, :);

    % 萤火虫移动
    for i = 1:fa_n
        for j = 1:fa_n
            if fa_light(i) > fa_light(j)
                r    = norm(fa_zn(i,:) - fa_zn(j,:));
                beta = (1 - fa_betamin) * exp(-fa_gamma * r^2) + fa_betamin;
                fa_zn(i,:) = fa_zn(i,:) * (1 - beta) + fa_zn(j,:) * beta ...
                             + fa_alpha * (rand(1, nVar) - 0.5);
                fa_zn(i,:) = min(max(fa_zn(i,:), lb), ub);
                fa_light(i) = fitness_fn(fa_zn(i,:), data, weights);
            end
        end
    end

    % 记录最优
    [cur_min, cur_idx] = min(fa_light);
    if cur_min < fa_bestScore
        fa_bestScore = cur_min;
        fa_zbest     = fa_zn(cur_idx, :);
    end
    fa_curve(k) = fa_bestScore;

    if mod(k, 50) == 0
        fprintf('  FA 第 %d 代，最优适应度: %.4f\n', k, fa_bestScore);
    end
end

fa_zbest(fa_zbest < 0.01) = 0.01;
X_FA   = ceil(fa_zbest .* data.P);
eID_FA = zeros(1, nVar);
for i = 1:nVar
    eID_FA(i) = data.DFenPei{i}(X_FA(i)+1);
end
fprintf('FA 完成，最优适应度: %.4f\n', fa_bestScore);

% ======================================================================
%  4. 差异统计（以 CS 为基准）
% ======================================================================
diff_ABC = (eID_ABC ~= eID_CS);
diff_FA  = (eID_FA  ~= eID_CS);

fprintf('\n========== 差异统计 ==========\n');
fprintf('灵活分配点总数         : %d\n', nVar);
fprintf('ABC 与 CS 不同         : %d 条 (%.1f%%)\n', sum(diff_ABC), 100*mean(diff_ABC));
fprintf('FA  与 CS 不同         : %d 条 (%.1f%%)\n', sum(diff_FA),  100*mean(diff_FA));

% ======================================================================
%  图1 - CS 收敛曲线
% ======================================================================
figure('Color','w','Name','CS Convergence','Position',[50,550,560,380]);
plot(cs_curve,'LineWidth',2,'Color',[0.85,0.33,0.10]);
grid on; box on;
xlabel('Iteration','FontWeight','bold');
ylabel('Best Fitness','FontWeight','bold');
title('CS Convergence Curve','FontSize',12,'FontWeight','bold');
xlim([1, cs_MaxIter]);

% ======================================================================
%  图2 - ABC 收敛曲线
% ======================================================================
figure('Color','w','Name','ABC Convergence','Position',[630,550,560,380]);
plot(abc_curve,'LineWidth',2,'Color',[0.55,0.10,0.90]);
grid on; box on;
xlabel('Iteration','FontWeight','bold');
ylabel('Best Fitness','FontWeight','bold');
title('ABC Convergence Curve','FontSize',12,'FontWeight','bold');
xlim([1, abc_MaxCycle]);

% ======================================================================
%  图3 - FA 收敛曲线
% ======================================================================
figure('Color','w','Name','FA Convergence','Position',[50,100,560,380]);
plot(fa_curve,'LineWidth',2,'Color',[0.42,0.78,0.60]);
grid on; box on;
xlabel('Iteration','FontWeight','bold');
ylabel('Best Fitness','FontWeight','bold');
title('FA Convergence Curve','FontSize',12,'FontWeight','bold');
xlim([1, fa_MaxGen]);

% ======================================================================
%  图4 - 三算法叠加对比地图（核心输出）
% ======================================================================
figure('Color','w','Name','CS / ABC / FA Comparison Map','Position',[630,50,1000,900]);
hold on; box on;

% 道路底图
if isfield(data,'road')
    for i = 1:length(data.road)
        plot(data.road{i}(:,1)-offset_x, data.road{i}(:,2)-offset_y, ...
             'Color',[0.88,0.88,0.88],'LineWidth',0.5);
    end
end

% --- CS 固定分配线（红色，所有算法共用底图） ---
if ~isempty(data.FID)
    for i = 1:size(data.FID,1)
        line([house_x(data.FID(i,1)), binan_x(data.FID(i,2))], ...
             [house_y(data.FID(i,1)), binan_y(data.FID(i,2))], ...
             'Color',[0.90, 0.10, 0.10, 0.45],'LineWidth',0.8);
    end
end

% --- CS 灵活分配线（红色，所有点，先铺底） ---
for i = 1:nVar
    hIdx = data.DFenPei{i}(1);
    line([house_x(hIdx), binan_x(eID_CS(i))], ...
         [house_y(hIdx), binan_y(eID_CS(i))], ...
         'Color',[0.90, 0.10, 0.10, 0.45],'LineWidth',0.8);
end

% --- ABC 与 CS 不同的分配线（紫色） ---
h_abc_diff = [];
for i = find(diff_ABC)
    hIdx = data.DFenPei{i}(1);
    h = line([house_x(hIdx), binan_x(eID_ABC(i))], ...
             [house_y(hIdx), binan_y(eID_ABC(i))], ...
             'Color',[0.55, 0.10, 0.90, 0.35],'LineWidth',1.0);
    if isempty(h_abc_diff), h_abc_diff = h; end
end

% --- FA 与 CS 不同的分配线（绿色） ---
h_fa_diff = [];
for i = find(diff_FA)
    hIdx = data.DFenPei{i}(1);
    h = line([house_x(hIdx), binan_x(eID_FA(i))], ...
             [house_y(hIdx), binan_y(eID_FA(i))], ...
             'Color',[0.42, 0.78, 0.60, 0.65],'LineWidth',1.0);
    if isempty(h_fa_diff), h_fa_diff = h; end
end

% --- 节点（最上层） ---
h_res = scatter(house_x, house_y, 9,  [0.00,0.18,0.60], 'filled');
h_shl = scatter(binan_x, binan_y, 80, 'g', '^', 'filled', ...
                'MarkerEdgeColor','k','LineWidth',0.8);

% --- 格式化 ---
axis equal; axis tight; grid on;
ax = gca;
ax.LooseInset  = ax.TightInset;
ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
ax.FontSize    = 11;  ax.FontWeight = 'bold';
margin = 100;
xlim([min(house_x)-margin, max(house_x)+margin]);
ylim([min(house_y)-margin, max(house_y)+margin]);

title('CS / ABC / FA Allocation Comparison Map', ...
      'FontSize',14,'FontWeight','bold');
xlabel('X Coordinate Offset (m)','FontWeight','bold');
ylabel('Y Coordinate Offset (m)','FontWeight','bold');

% --- 图例 ---
h_cs_leg  = line(NaN,NaN,'Color',[0.90,0.10,0.10,0.80],'LineWidth',2.0);
h_abc_leg = line(NaN,NaN,'Color',[0.55,0.10,0.90,0.50],'LineWidth',2.0);
h_fa_leg  = line(NaN,NaN,'Color',[0.42,0.78,0.60,0.80],'LineWidth',2.0);

legend([h_shl, h_res, h_cs_leg, h_abc_leg, h_fa_leg], ...
    {'Shelter', ...
     'Residential', ...
     'CS allocation (baseline)', ...
     'ABC allocation (differs from CS)', ...
     'FA  allocation (differs from CS)'}, ...
    'Location','northeast','FontSize',7,'Box','on');

fprintf('\n全部完成！共耗时 %.1f 秒。\n', toc);
fprintf('\n图例说明:\n');
fprintf('  红色线   = CS 全部分配（底图基准）\n');
fprintf('  紫色粗线 = ABC 与 CS 结果不同的分配连线\n');
fprintf('  绿色粗线 = FA  与 CS 结果不同的分配连线\n');
fprintf('  绿色三角 = 避难场所\n');
fprintf('  蓝色圆点 = 住宅点位\n');

% ======================================================================
%  本地函数区
% ======================================================================

% ---------- 共用单目标适应度函数 ----------
function score = fitness_fn(x, S, w)
    x(x < 0.01) = 0.01;
    X = ceil(x .* S.P);
    total_dist = S.alldis;
    Y = S.YFenPei;
    for i = 1:length(X)
        hID = S.DFenPei{i}(1);
        eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12;
    end
    score = w.w1 * total_dist + w.w2 * var(Y);
end