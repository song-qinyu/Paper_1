%% DE 疏散分配优化 - 自动评价指标版
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
house_x = raw_x - offset_x;
house_y = raw_y - offset_y;
binan_x = data.binan(:,1) - offset_x;
binan_y = data.binan(:,2) - offset_y;

% 适配字段
if exist('dis','var'), data.dis = dis; end
data.P = cellfun(@(x) length(x)-1, data.DFenPei);

% 预处理固定分配
data.alldis_fixed = 0;
data.YFenPei_init = zeros(1, size(data.binan, 1));
data.FID = []; 
for k = 1:length(B)
    if length(B{k}) == 1
        targetBinan = B{k};
        data.YFenPei_init(targetBinan) = data.YFenPei_init(targetBinan) + 12;
        data.FID = [data.FID; k, targetBinan];
        data.alldis_fixed = data.alldis_fixed + data.dis(k, targetBinan);
    end
end

% ========================== 2. DE 参数设置 ==========================
nVar = length(data.DFenPei);  
popSize = 50;       
maxGen = 200;       
F = 0.5;            % 变异因子
CR = 0.9;           % 交叉概率
weights.w1 = 0.001; weights.w2 = 1.0;   

% ========================== 3. 执行 DE 进化 ==========================
fprintf('DE 优化启动，正在计算指标...\n');
DEpop = rand(popSize, nVar);
fitness = zeros(popSize, 1);
for i = 1:popSize
    fitness(i) = fitness_function(DEpop(i,:), data, weights);
end

[bestScore, bestIdx] = min(fitness);
zbest = DEpop(bestIdx, :);
cg_curve = zeros(1, maxGen);

for g = 1:maxGen
    for i = 1:popSize
        % 变异操作: r1 + F*(r2 - r3)
        idxs = randperm(popSize, 3);
        while any(idxs == i), idxs = randperm(popSize, 3); end
        v = DEpop(idxs(1),:) + F * (DEpop(idxs(2),:) - DEpop(idxs(3),:));
        
        % 交叉操作
        jRand = randi(nVar);
        t = rand(1, nVar) < CR;
        t(jRand) = 1;
        u = DEpop(i,:);
        u(t) = v(t);
        
        % 边界检查与评估
        u(u>1)=1; u(u<0.01)=0.01;
        f_u = fitness_function(u, data, weights);
        
        % 选择操作
        if f_u < fitness(i)
            DEpop(i,:) = u;
            fitness(i) = f_u;
            if f_u < bestScore
                bestScore = f_u;
                zbest = u;
            end
        end
    end
    cg_curve(g) = bestScore;
end

% ========================== 4. 计算评价指标 (Evaluation Metrics) ==========================
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
SD = 0; % 单次运行设为0，多次运行取 std(BTV_history)

% ========================== 5. 输出结果面板 ==========================
fprintf('\n==============================================\n');
fprintf('   DE Algorithm Performance Metrics (sj5.mat)\n');
fprintf('==============================================\n');
fprintf('Algorithm: DE (Differential Evolution)\n');
fprintf('TED (Total Distance):   %.2f m\n', TED);
fprintf('ATD (Avg Distance):     %.2f m\n', ATD);
fprintf('MID (Max Distance):     %.2f m\n', MID);
fprintf('SUR (Utilization Rate): %.2f %%\n', SUR);
fprintf('BTV (Best Fitness):     %.6f\n', BTV);
fprintf('MET (Exec Time):        %.4f s\n', MET);
fprintf('CG  (Conv. Gen):        %d\n', CG);
fprintf('SD  (Stability):        %.4f\n', SD);
fprintf('==============================================\n');

% --- 快速绘图验证 ---
figure('Color','w');
subplot(1,2,1); plot(cg_curve, 'LineWidth', 2); title('DE Convergence');
subplot(1,2,2); hold on;
scatter(house_x, house_y, 5, [0.5 0.5 0.5]);
scatter(binan_x, binan_y, 50, 'r', '^', 'filled');
title('DE Final Allocation'); axis equal;

% ========================== 适应度函数 ==========================
function score = fitness_function(x, S, w)
    x(x < 0.01) = 0.01;
    X = ceil(x .* S.P); 
    total_dist = S.alldis_fixed;
    Y = S.YFenPei_init; 
    for i = 1:length(X)
        hID = S.DFenPei{i}(1); eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12; 
    end
    score = w.w1 * total_dist + w.w2 * var(Y); 
end