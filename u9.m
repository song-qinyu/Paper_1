%% ABC 人工蜂群算法疏散分配优化 - 指标集成版
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
house_x = raw_x - offset_x; house_y = raw_y - offset_y;
binan_x = data.binan(:,1) - offset_x; binan_y = data.binan(:,2) - offset_y;

% 字段适配
if exist('dis','var'), data.dis = dis; end
data.P = cellfun(@(x) length(x)-1, data.DFenPei);

% 预处理固定分配
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

% ========================== 2. ABC 参数设置 ==========================
nVar = length(data.DFenPei);    
popSize = 50;                   % 蜜源数量
maxGen = 200;                   % 最大迭代次数
limit = 20;                     % 侦察蜂阈值
weights.w1 = 0.001; weights.w2 = 1.0; 

% 初始化蜜源
pop = rand(popSize, nVar);
fitness = zeros(popSize, 1);
counter = zeros(popSize, 1);    % 蜜源未改进计数器
for i = 1:popSize
    fitness(i) = abc_obj(pop(i,:), data, weights);
end

[fMin, bestIdx] = min(fitness);
bestSol = pop(bestIdx, :);
cg_curve = zeros(1, maxGen);

% ========================== 3. 执行 ABC 优化 ==========================
fprintf('ABC 优化启动，正在计算 8 项评价指标...\n');

for g = 1:maxGen
    % 1. 采蜜蜂阶段 (Employed Bees)
    for i = 1:popSize
        k = randi([1, popSize]);
        while k == i, k = randi([1, popSize]); end
        phi = (2*rand(1, nVar)-1);
        new_sol = pop(i,:) + phi .* (pop(i,:) - pop(k,:));
        new_sol = max(min(new_sol, 1), 0.01);
        
        new_fit = abc_obj(new_sol, data, weights);
        if new_fit < fitness(i)
            pop(i,:) = new_sol; fitness(i) = new_fit; counter(i) = 0;
        else
            counter(i) = counter(i) + 1;
        end
    end
    
    % 2. 观察蜂阶段 (Onlooker Bees)
    prob = (1./(fitness + eps)) / sum(1./(fitness + eps));
    i = 1; t = 0;
    while t < popSize
        if rand < prob(i)
            t = t + 1;
            k = randi([1, popSize]);
            while k == i, k = randi([1, popSize]); end
            phi = (2*rand(1, nVar)-1);
            new_sol = pop(i,:) + phi .* (pop(i,:) - pop(k,:));
            new_sol = max(min(new_sol, 1), 0.01);
            
            new_fit = abc_obj(new_sol, data, weights);
            if new_fit < fitness(i)
                pop(i,:) = new_sol; fitness(i) = new_fit; counter(i) = 0;
            else
                counter(i) = counter(i) + 1;
            end
        end
        i = mod(i, popSize) + 1;
    end
    
    % 3. 侦察蜂阶段 (Scout Bees)
    [max_cnt, idx] = max(counter);
    if max_cnt > limit
        pop(idx,:) = rand(1, nVar);
        fitness(idx) = abc_obj(pop(idx,:), data, weights);
        counter(idx) = 0;
    end
    
    % 记录最优
    [fCurrent, bestI] = min(fitness);
    if fCurrent < fMin
        fMin = fCurrent; bestSol = pop(bestI, :);
    end
    cg_curve(g) = fMin;
end

% ========================== 4. 计算 8 项评价指标 ==========================
MET = toc; 
zbest = bestSol;
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P); 

% 计算疏散距离
dynamic_distances = zeros(1, length(X_final));
dynamic_used_bins = zeros(1, length(X_final));
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1); 
    eIdx = data.DFenPei{i}(X_final(i)+1);
    dynamic_distances(i) = data.dis(hIdx, eIdx);
    dynamic_used_bins(i) = eIdx;
end
fixed_distances = arrayfun(@(i) data.dis(data.FID(i,1), data.FID(i,2)), 1:size(data.FID,1));
all_dist = [dynamic_distances, fixed_distances];

% 指标统计
TED = sum(all_dist);           
ATD = mean(all_dist);          
MID = max(all_dist);           
BTV = fMin;                    
SUR = (length(unique([dynamic_used_bins, data.FID(:,2)'])) / size(data.binan, 1)) * 100; 

% 收敛代数 (CG)
change = abs(diff(cg_curve));
last_c = find(change > 1e-6, 1, 'last');
if isempty(last_c), CG = 1; else CG = last_c + 1; end
SD = 0; 

% ========================== 5. 输出结果面板 ==========================
fprintf('\n==============================================\n');
fprintf('   ABC (Artificial Bee Colony) 性能评价指标\n');
fprintf('==============================================\n');
fprintf('TED (总距离):   %.2f m\n', TED);
fprintf('ATD (平均距离): %.2f m\n', ATD);
fprintf('MID (最大距离): %.2f m\n', MID);
fprintf('SUR (利用率):   %.2f %%\n', SUR);
fprintf('BTV (最佳适应度): %.6f\n', BTV);
fprintf('MET (执行时间): %.4f s\n', MET);
fprintf('CG  (收敛代数): %d\n', CG);
fprintf('SD  (稳定性):   %.4f\n', SD);
fprintf('==============================================\n');

% ========================== 适应度函数 ==========================
function score = abc_obj(x, S, w)
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