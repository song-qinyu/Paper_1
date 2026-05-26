%% PSO 疏散分配优化 - 自动评价指标版
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

% ========================== 2. PSO 参数设置 ==========================
nVar = length(data.DFenPei);  
popSize = 80;       
maxGen = 200;       
w = 0.8; c1 = 1.5; c2 = 1.5;  
weights.w1 = 0.001; weights.w2 = 1.0;   

% ========================== 3. 执行 PSO 进化 ==========================
fprintf('PSO 优化启动，正在计算指标...\n');
particlePos = rand(popSize, nVar);
particleVel = zeros(popSize, nVar);
pBestPos = particlePos;
pBestScore = zeros(popSize, 1);

for i = 1:popSize
    pBestScore(i) = fitness_function(particlePos(i,:), data, weights);
end

[gBestScore, gIdx] = min(pBestScore);
gBestPos = pBestPos(gIdx, :);
cg_curve = zeros(1, maxGen);

for g = 1:maxGen
    for i = 1:popSize
        % 速度更新
        particleVel(i,:) = w*particleVel(i,:) + c1*rand(1,nVar).*(pBestPos(i,:)-particlePos(i,:)) + ...
                           c2*rand(1,nVar).*(gBestPos-particlePos(i,:));
        % 位置更新
        particlePos(i,:) = particlePos(i,:) + particleVel(i,:);
        
        % 边界处理
        particlePos(i, particlePos(i,:)>1) = 1;
        particlePos(i, particlePos(i,:)<0.01) = 0.01;
        
        % 评估
        currentScore = fitness_function(particlePos(i,:), data, weights);
        if currentScore < pBestScore(i)
            pBestScore(i) = currentScore;
            pBestPos(i,:) = particlePos(i,:);
            if pBestScore(i) < gBestScore
                gBestScore = pBestScore(i);
                gBestPos = pBestPos(i,:);
            end
        end
    end
    cg_curve(g) = gBestScore;
end

% ========================== 4. 计算评价指标 (Evaluation Metrics) ==========================
zbest = gBestPos;
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P); 

% 指标 1: BTV (Best Fitness Value)
BTV = gBestScore;

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

% ========================== 5. 输出结果 ==========================
fprintf('\n==============================================\n');
fprintf('   PSO Algorithm Performance Metrics (sj5.mat)\n');
fprintf('==============================================\n');
fprintf('Algorithm: PSO (Particle Swarm Optimization)\n');
fprintf('TED: %.2f m\n', TED);
fprintf('ATD: %.2f m\n', ATD);
fprintf('MID: %.2f m\n', MID);
fprintf('SUR: %.2f %%\n', SUR);
fprintf('BTV: %.6f\n', BTV);
fprintf('MET: %.4f s\n', MET);
fprintf('CG:  %d\n', CG);
fprintf('SD:  %.4f\n', SD);
fprintf('==============================================\n');

% ========================== 适应度函数 ==========================
function score = fitness_function(x, S, w)
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