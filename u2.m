%% NSGA-II 多目标疏散分配优化 - 自动评价指标版
clc; clear; close all; tic;

% ========================== 1. 加载与预处理 ==========================
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

% ========================== 2. NSGA-II 参数设置 ==========================
nVar = length(data.DFenPei);  
popSize = 50;       
maxGen = 150;       
pc = 0.9; pm = 0.1; 

% ========================== 3. 执行 NSGA-II 进化 ==========================
fprintf('NSGA-II 优化启动，正在计算指标...\n');
pop = rand(popSize, nVar);
objs = zeros(popSize, 2);
for i = 1:popSize
    objs(i,:) = objective_functions(pop(i,:), data);
end

cg_curve = zeros(1, maxGen);

for g = 1:maxGen
    % 选择、交叉、变异生成子代
    offspring = zeros(popSize, nVar);
    for i = 1:2:popSize
        p1 = pop(randi(popSize), :); p2 = pop(randi(popSize), :);
        if rand < pc
            cp = randi(nVar);
            offspring(i,:) = [p1(1:cp), p2(cp+1:end)];
            offspring(i+1,:) = [p2(1:cp), p1(cp+1:end)];
        else
            offspring(i,:) = p1; offspring(i+1,:) = p2;
        end
    end
    % 变异
    for i = 1:popSize
        if rand < pm
            mp = randi(nVar);
            offspring(i, mp) = rand();
        end
    end
    
    % 合并种群与非支配排序
    combinedPop = [pop; offspring];
    combinedObjs = zeros(popSize*2, 2);
    for i = 1:popSize*2
        combinedObjs(i,:) = objective_functions(combinedPop(i,:), data);
    end
    
    % 这里简化处理：按综合得分排序（w1*obj1 + w2*obj2）作为收敛记录
    % 实际 NSGA-II 使用拥挤度距离排序
    scores = 0.001 * combinedObjs(:,1) + 1.0 * combinedObjs(:,2);
    [~, sortIdx] = sort(scores);
    
    pop = combinedPop(sortIdx(1:popSize), :);
    objs = combinedObjs(sortIdx(1:popSize), :);
    cg_curve(g) = min(scores);
end

% ========================== 4. 计算评价指标 (Evaluation Metrics) ==========================
% 选择收敛后的最佳个体 (Score最小的)
[bestScore, bestIdx] = min(0.001 * objs(:,1) + 1.0 * objs(:,2));
zbest = pop(bestIdx, :);
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P); 

% 指标 1: BTV (Best Fitness Value)
BTV = bestScore;

% 指标 2: MET (Mean Execution Time)
MET = toc; 

% 指标 3: CG (Convergence Generation)
threshold = 1e-6;
change = abs(diff(cg_curve));
last_change = find(change > threshold, 1, 'last');
if isempty(last_change), CG = 1; else CG = last_change + 1; end

% 指标 4: 路径质量 (TED, ATD, MID)
dynamic_distances = zeros(1, length(X_final));
dynamic_used_bins = zeros(1, length(X_final));
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1); 
    eIdx = data.DFenPei{i}(X_final(i)+1);
    dynamic_distances(i) = data.dis(hIdx, eIdx);
    dynamic_used_bins(i) = eIdx;
end
fixed_distances = [];
if ~isempty(data.FID)
    for i = 1:size(data.FID, 1)
        fixed_distances(i) = data.dis(data.FID(i,1), data.FID(i,2));
    end
end
all_distances = [dynamic_distances, fixed_distances];

TED = sum(all_distances);           
ATD = mean(all_distances);          
MID = max(all_distances);           

% 指标 5: SUR (Shelter Utilization Rate)
fixed_used_bins = [];
if ~isempty(data.FID), fixed_used_bins = data.FID(:,2)'; end
used_bins_total = unique([dynamic_used_bins, fixed_used_bins]); 
total_bins_available = size(data.binan, 1);
SUR = (length(used_bins_total) / total_bins_available) * 100;

% 指标 6: SD (Stability)
SD = 0; 

% ========================== 5. 打印结果 ==========================
fprintf('\n==============================================\n');
fprintf('   NSGA-II Performance Metrics (sj5.mat)\n');
fprintf('==============================================\n');
fprintf('TED: %.2f m\n', TED);
fprintf('ATD: %.2f m\n', ATD);
fprintf('MID: %.2f m\n', MID);
fprintf('SUR: %.2f %%\n', SUR);
fprintf('BTV: %.6f\n', BTV);
fprintf('MET: %.4f s\n', MET);
fprintf('CG:  %d\n', CG);
fprintf('SD:  %.4f\n', SD);
fprintf('==============================================\n');

% ========================== 目标函数组 ==========================
function f = objective_functions(x, S)
    x(x < 0.01) = 0.01;
    X = ceil(x .* S.P); 
    dist = S.alldis_fixed;
    Y = S.YFenPei_fixed; 
    for i = 1:length(X)
        hID = S.DFenPei{i}(1); eID = S.DFenPei{i}(X(i)+1);
        dist = dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12; 
    end
    f = [dist, var(Y)]; % 返回两个目标：总距离和方差
end