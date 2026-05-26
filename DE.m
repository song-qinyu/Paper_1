%% 疏散分配优化全套改进版 (DE/GA/PSO 通用逻辑)
clc; clear; close all;

% ========================== 1. 加载与适配数据 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat');
end

% 确保核心矩阵存在
if exist('dis','var'), data.dis = dis; end
% P 代表每个点在 DFenPei 中对应的可选避难所数量
data.P = cellfun(@(x) length(x)-1, data.DFenPei);

% --- 关键改进：预处理所有点的分配状态 ---
numTotalHomes = size(data.start, 1);
data.YFenPei = zeros(1, size(data.binan, 1));
data.alldis = 0;
data.FID = []; % 存储固定分配点 [宅基地ID, 避难点ID]

% 识别固定分配点（只有一个选择的点）
for k = 1:length(B)
    if length(B{k}) == 1
        targetBinan = B{k};
        data.YFenPei(targetBinan) = data.YFenPei(targetBinan) + 12; 
        data.FID = [data.FID; k, targetBinan];
        data.alldis = data.alldis + data.dis(k, targetBinan);
    end
end

% ========================== 2. 算法参数设置 ==========================
nVar = length(data.DFenPei);  % 待优化的点数
popSize = 60;
maxGen = 150;
weights.w1 = 0.001; weights.w2 = 1.0; 
userData.vMax = 1.2; 

% ========================== 3. 执行优化 (以 DE 为例) ==========================
fprintf('正在搜索最优分配方案...\n');
% 初始化：直接在 [0,1] 空间随机，不再依赖栅格地图 G
pop = rand(popSize, nVar);
fitness = zeros(popSize, 1);
for i = 1:popSize
    fitness(i) = fitness_function(pop(i,:), data, weights);
end

[bestScore, bestIdx] = min(fitness);
zbest = pop(bestIdx, :);
cg_curve = zeros(1, maxGen);

% 差分进化 (DE) 迭代核心
for g = 1:maxGen
    for i = 1:popSize
        % 变异
        A = randperm(popSize, 4); A(A==i) = [];
        v = pop(A(1),:) + 0.5 * (pop(A(2),:) - pop(A(3),:));
        % 交叉
        j0 = randi(nVar);
        for j = 1:nVar
            if rand < 0.3 || j == j0
                target_pos = v(j);
                % 边界约束
                if target_pos > 1, target_pos = 1; end
                if target_pos < 0, target_pos = 0; end
                pop(i,j) = target_pos;
            end
        end
        % 更新
        f_new = fitness_function(pop(i,:), data, weights);
        if f_new < fitness(i)
            fitness(i) = f_new;
            if f_new < bestScore
                bestScore = f_new;
                zbest = pop(i,:);
            end
        end
    end
    cg_curve(g) = bestScore;
end

% ========================== 4. 核心改进：完整绘图逻辑 ==========================
fprintf('===== 优化完成，正在绘制全量连接图 =====\n');

% 解码最优方案
zbest(zbest < 0.01) = 0.01;
X_optimized = ceil(zbest .* data.P); 

% 统计所有点的疏散时间 (用于柱状图)
evacuationTimes = zeros(numTotalHomes, 1);

figure('Color', 'w', 'Name', '全量疏散分配地图');
hold on;

% 1. 绘制道路底图
if isfield(data, 'road')
    for i = 1:length(data.road)
        plot(data.road{i}(:,1), data.road{i}(:,2), 'Color', [0.9 0.9 0.9], 'LineWidth', 0.5);
    end
end

% 2. 绘制【固定分配】的连线 (之前可能漏掉的部分)
for i = 1:size(data.FID, 1)
    hIdx = data.FID(i,1);
    eIdx = data.FID(i,2);
    plot([data.start(hIdx,1), data.binan(eIdx,1)], [data.start(hIdx,2), data.binan(eIdx,2)], ...
        'Color', [0.4660 0.6740 0.1880], 'LineWidth', 0.5); % 绿色线表示固定分配
    evacuationTimes(hIdx) = (data.dis(hIdx, eIdx) * 0.01344) / userData.vMax;
end

% 3. 绘制【优化分配】的连线
for i = 1:length(X_optimized)
    hIdx = data.DFenPei{i}(1);
    eIdx = data.DFenPei{i}(X_optimized(i)+1);
    plot([data.start(hIdx,1), data.binan(eIdx,1)], [data.start(hIdx,2), data.binan(eIdx,2)], ...
        'Color', [0 0.4470 0.7410], 'LineWidth', 0.5); % 蓝色线表示优化分配
    evacuationTimes(hIdx) = (data.dis(hIdx, eIdx) * 0.01344) / userData.vMax;
end

% 4. 绘制点位图层 (放在线上面)
h_bin = scatter(data.binan(:,1), data.binan(:,2), 80, 'g^', 'filled', 'MarkerEdgeColor', 'k');
h_home = scatter(data.start(:,1), data.start(:,2), 20, 'ro', 'filled');

title(['全量疏散分配方案 (已确保 ', num2str(numTotalHomes), ' 个点全部连接)']);
legend([h_bin, h_home], '避难点', '宅基地');
axis equal; grid on;

% --- 辅助验证：统计未连接点 ---
unconnected = find(evacuationTimes == 0);
if isempty(unconnected)
    fprintf('验证通过：所有 %d 个点均已成功建立连接线。\n', numTotalHomes);
else
    fprintf('警告：仍有 %d 个点未连接，请检查数据完整性。\n', length(unconnected));
end

% --- 收敛曲线与时间分布图 ---
figure('Color','w'); plot(cg_curve, 'LineWidth', 2); title('算法收敛曲线');
figure('Color','w'); bar(evacuationTimes, 'EdgeColor', 'none'); 
hold on; line([0 numTotalHomes],[900 900],'Color','r','LineStyle','--');
title('各宅基地疏散时间统计'); ylabel('时间 (s)');

% ========================== 适应度函数 ==========================
function score = fitness_function(x, S, w)
    x(x < 0.01) = 0.01;
    X = ceil(x .* S.P); 
    total_dist = S.alldis;
    Y = S.YFenPei; 
    for i = 1:length(X)
        hID = S.DFenPei{i}(1); eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12; 
    end
    score = w.w1 * total_dist + w.w2 * var(Y); 
end