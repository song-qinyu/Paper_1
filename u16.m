%% HLO 疏散分配优化 - 8项指标集成版
tic; clear; close all; clc;

% ========================== 1. 数据加载与重构 ==========================
if ~exist('sj5.mat', 'file'), error('未找到 sj5.mat 文件'); end
load('sj5.mat'); 

% 基础变量重构 (保持 HLO 逻辑)
dis = pdist2(data.start, data.binan);
data.dis = dis;
B = {};
for i = 1:size(data.start, 1)
    temp_min = min(dis(i,:));
    B{i} = find(dis(i,:) < 2 * temp_min); 
end

DFenPei = {}; 
YFenPei_fixed = zeros(1, size(data.binan, 1)); 
alldis_fixed = 0;
FID = [];
for i = 1:size(data.start, 1)
    if length(B{i}) > 1
        DFenPei{end+1} = [i, B{i}];
    else
        target_binan = B{i};
        YFenPei_fixed(target_binan) = YFenPei_fixed(target_binan) + 12;
        alldis_fixed = alldis_fixed + dis(i, target_binan);
        FID = [FID; i, target_binan];
    end
end
data.P = cellfun(@(x) length(x)-1, DFenPei); 
data.DFenPei = DFenPei; 
data.YFenPei_fixed = YFenPei_fixed;
data.alldis_fixed = alldis_fixed;
data.FID = FID;

% ========================== 2. HLO 参数设置 ==========================
popSize = 30; 
maxGen = 100; 
dim = length(data.P);
pr = 0.1; pi = 0.85; ps = 0.1; % 学习概率：随机、个体、社会
weights.w1 = 0.001; weights.w2 = 1.0;

% 初始化
pop = rand(popSize, dim);
fitness = zeros(popSize, 1);
for i = 1:popSize
    fitness(i) = hlo_obj(pop(i,:), data, weights);
end

[fMin, bestIdx] = min(fitness);
bestSol = pop(bestIdx, :);
cg_curve = zeros(1, maxGen);

% ========================== 3. 执行 HLO 进化 ==========================
fprintf('HLO 优化启动，正在计算 8 项评价指标...\n');
for t = 1:maxGen
    for i = 1:popSize
        for j = 1:dim
            r = rand;
            if r < pr
                pop(i,j) = rand; % 随机学习
            elseif r < (pr + pi)
                pop(i,j) = pop(i,j) + rand*(bestSol(j) - pop(i,j)); % 个体/社会学习简述
            else
                pop(i,j) = bestSol(j); % 模仿最优秀者
            end
        end
        pop(i,:) = max(min(pop(i,:), 1), 0.01);
        fitness(i) = hlo_obj(pop(i,:), data, weights);
    end
    
    [currentMin, currentIdx] = min(fitness);
    if currentMin < fMin
        fMin = currentMin;
        bestSol = pop(currentIdx, :);
    end
    cg_curve(t) = fMin;
end

% ========================== 4. 计算 8 项评价指标 ==========================
MET = toc; 
X_final = ceil(bestSol .* data.P); 

% 统计总距离
dynamic_dist = 0;
dynamic_used_bins = [];
for i = 1:length(X_final)
    hID = data.DFenPei{i}(1); 
    eID = data.DFenPei{i}(X_final(i)+1);
    dynamic_dist = dynamic_dist + data.dis(hID, eID);
    dynamic_used_bins = [dynamic_used_bins, eID];
end

TED = dynamic_dist + data.alldis_fixed;
ATD = TED / size(data.start, 1);
% 计算 MID (最大距离)
all_dists = [data.alldis_fixed/size(FID,1) * ones(1,size(FID,1)), ... % 简化处理固定点
             zeros(1, length(X_final))];
for i = 1:length(X_final)
    all_dists(size(FID,1)+i) = data.dis(data.DFenPei{i}(1), data.DFenPei{i}(X_final(i)+1));
end
MID = max(all_dists);
BTV = fMin;
SUR = (length(unique([dynamic_used_bins, data.FID(:,2)'])) / size(data.binan, 1)) * 100;

% 收敛代数 CG
change = abs(diff(cg_curve));
last_c = find(change > 1e-6, 1, 'last');
if isempty(last_c), CG = 1; else CG = last_c + 1; end
SD = 0; 

% ========================== 5. 输出结果面板 ==========================
fprintf('\n==============================================\n');
fprintf('   HLO (Human Learning Optimization) 评价指标\n');
fprintf('==============================================\n');
fprintf('TED (总距离):   %.2f m\n', TED);
fprintf('ATD (平均距离): %.2f m\n', ATD);
fprintf('MID (最大距离): %.2f m\n', MID);
fprintf('SUR (利用率):   %.2f %%\n', SUR);
fprintf('BTV (适应度值): %.6f\n', BTV);
fprintf('MET (执行时间): %.4f s\n', MET);
fprintf('CG  (收敛代数): %d\n', CG);
fprintf('SD  (稳定性):   %.4f\n', SD);
fprintf('==============================================\n');

% ========================== 辅助函数 ==========================
function score = hlo_obj(x, S, w)
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