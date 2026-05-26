%% SSA 疏散分配优化集成主程序 - 三图全功能版
clc; clear; close all;

% ========================== 1. 数据加载与预处理 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat');
end

if exist('dis','var'), data.dis = dis; end
data.P = cellfun(@(x) length(x)-1, data.DFenPei);
data.YFenPei = zeros(1, size(data.binan, 1));
data.alldis = 0;
data.FID = []; 
optimized_home_ids = cellfun(@(x) x(1), data.DFenPei);

for k = 1:size(data.start, 1)
    if ~ismember(k, optimized_home_ids)
        targetBinan = B{k}(1);
        data.YFenPei(targetBinan) = data.YFenPei(targetBinan) + 12; 
        data.FID = [data.FID; k, targetBinan];
        data.alldis = data.alldis + data.dis(k, targetBinan);
    end
end

% ========================== 2. 参数设置 ==========================
pop = 50;           
M = 100;            
dim = length(data.DFenPei); 
lb = 0.01;          
ub = 1.0;           

% ========================== 3. 执行 SSA 优化 ==========================
fprintf('SSA 算法启动，正在优化 %d 个点的分配方案...\n', dim);
[fMin, bestX, Convergence_curve] = SSA_Core(pop, M, lb, ub, dim, data);

% ========================== 4. 结果可视化 (3张图) ==========================
fprintf('正在生成结果分析图表...\n');
bestX(bestX < 0.01) = 0.01;
X_final = ceil(bestX .* data.P); 

% --- 初始化时间统计 ---
vMax = 1.2; 
numTotal = size(data.start, 1);
evacuationTimes = zeros(numTotal, 1);

% 【图 1: 全量地理分配地图】
figure('Color', 'w', 'Name', '地理分配地图'); hold on;
if isfield(data, 'road')
    for i = 1:length(data.road), plot(data.road{i}(:,1), data.road{i}(:,2), 'Color', [0.9 0.9 0.9]); end
end

% 绘制固定点连线并记录时间
for i = 1:size(data.FID, 1)
    hIdx = data.FID(i,1); eIdx = data.FID(i,2);
    plot([data.start(hIdx,1), data.binan(eIdx,1)], [data.start(hIdx,2), data.binan(eIdx,2)], 'g-', 'LineWidth', 0.5);
    evacuationTimes(hIdx) = (data.dis(hIdx, eIdx) * 0.01344) / vMax;
end
% 绘制优化点连线并记录时间
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1); eIdx = data.DFenPei{i}(X_final(i)+1);
    plot([data.start(hIdx,1), data.binan(eIdx,1)], [data.start(hIdx,2), data.binan(eIdx,2)], 'b-', 'LineWidth', 0.5);
    evacuationTimes(hIdx) = (data.dis(hIdx, eIdx) * 0.01344) / vMax;
end

scatter(data.binan(:,1), data.binan(:,2), 80, 'g^', 'filled');
scatter(data.start(:,1), data.start(:,2), 20, 'ro', 'filled');
title(['SSA 方案：', num2str(numTotal), ' 个点已全部连接']); axis equal; grid on;

% 【图 2: 收敛曲线图】
figure('Color','w', 'Name', '算法收敛分析'); 
plot(Convergence_curve, 'r', 'LineWidth', 2);
xlabel('迭代次数'); ylabel('适应度值'); title('SSA 收敛过程'); grid on;

% 【图 3: 疏散时间分布图】(这是之前漏掉的)
figure('Color','w', 'Name', '疏散时间统计');
bar(evacuationTimes, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'none');
hold on; 
% 绘制 900s 警戒线
line([0, numTotal], [900, 900], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
xlabel('宅基地编号'); ylabel('疏散时间 (秒)');
title('各宅基地疏散时间分布（红色虚线为900s限制）');
grid on;

% ========================== 核心算法函数 (保持不变) ==========================
function [fMin, bestX, curve] = SSA_Core(pop, M, c, d, dim, S)
    P_percent = 0.2; pNum = round(pop * P_percent);
    lb = c * ones(1, dim); ub = d * ones(1, dim);
    x = lb + (ub - lb) .* rand(pop, dim);
    fit = zeros(1, pop);
    for i = 1:pop, fit(i) = internal_fitness(x(i,:), S); end
    [fMin, bestI] = min(fit); bestX = x(bestI, :); pFit = fit; pX = x; curve = zeros(1, M);
    for t = 1:M
        [~, sortIndex] = sort(pFit); [fmax, B_idx] = max(pFit); worse = x(B_idx, :);
        r2 = rand;
        for i = 1:pNum
            if r2 < 0.8, x(sortIndex(i),:) = pX(sortIndex(i),:) .* exp(-i/(rand*M));
            else, x(sortIndex(i),:) = pX(sortIndex(i),:) + randn(1,dim); end
            x(sortIndex(i),:) = max(min(x(sortIndex(i),:), ub), lb);
            fit(sortIndex(i)) = internal_fitness(x(sortIndex(i),:), S);
        end
        [~, bestII] = min(fit); bestXX = x(bestII, :);
        for i = (pNum+1):pop
            if i > pop/2, x(sortIndex(i),:) = randn * exp((worse - pX(sortIndex(i),:)) / (i^2));
            else, A = floor(rand(1, dim) * 2) * 2 - 1; L = A .* ( (A * A')^-1 );
                x(sortIndex(i),:) = bestXX + abs(pX(sortIndex(i),:) - bestXX) .* L; end
            x(sortIndex(i),:) = max(min(x(sortIndex(i),:), ub), lb);
            fit(sortIndex(i)) = internal_fitness(x(sortIndex(i),:), S);
        end
        arand = randperm(pop); b_idx = arand(1:min(15, pop));
        for j = 1:length(b_idx)
            if pFit(b_idx(j)) > fMin, x(b_idx(j),:) = bestX + randn(1,dim) .* abs(pX(b_idx(j),:) - bestX);
            else, x(b_idx(j),:) = pX(b_idx(j),:) + (2*rand-1) * abs(pX(b_idx(j),:) - worse) / (pFit(b_idx(j)) - fmax + 1e-50); end
            x(b_idx(j),:) = max(min(x(b_idx(j),:), ub), lb);
            fit(b_idx(j)) = internal_fitness(x(b_idx(j),:), S);
        end
        for i = 1:pop
            if fit(i) < pFit(i), pFit(i) = fit(i); pX(i,:) = x(i,:); end
            if pFit(i) < fMin, fMin = pFit(i); bestX = pX(i,:); end
        end
        curve(t) = fMin;
    end
end

function score = internal_fitness(x, S)
    w1 = 0.001; w2 = 1.0;
    x(x < 0.01) = 0.01; X = ceil(x .* S.P); 
    total_dist = S.alldis; Y = S.YFenPei; 
    for i = 1:length(X)
        hID = S.DFenPei{i}(1); eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12; 
    end
    score = w1 * total_dist + w2 * var(Y); 
end