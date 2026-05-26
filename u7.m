%% CS 布谷鸟算法疏散分配优化 - 指标集成版
clear; close all; clc; tic;

% ========================== 1. 加载与适配数据 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat');
end

% 核心适配：确保 dis 变量被存入 data 结构体
if exist('dis','var'), data.dis = dis; end
data.P = cellfun(@(x) length(x)-1, data.DFenPei);

% 坐标偏移处理
raw_x = data.start(:,1); raw_y = data.start(:,2);
offset_x = min(raw_x); offset_y = min(raw_y);
house_x = raw_x - offset_x; house_y = raw_y - offset_y;
binan_x = data.binan(:,1) - offset_x; binan_y = data.binan(:,2) - offset_y;

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

% ========================== 2. CS 参数设置 ==========================
nVar = length(data.DFenPei);    
popSize = 30;                   
maxGen = 150;                   
Pa = 0.25;                      % 发现概率
weights.w1 = 0.001; weights.w2 = 1.0; 

% 初始化巢穴
nest = rand(popSize, nVar);
fitness = zeros(popSize, 1);
for i = 1:popSize
    fitness(i) = cs_obj(nest(i,:), data, weights);
end

[fMin, bestIdx] = min(fitness);
bestNest = nest(bestIdx, :);
cg_curve = zeros(1, maxGen);

% ========================== 3. 执行 CS 进化 ==========================
fprintf('CS 优化启动，正在计算指标...\n');
for t = 1:maxGen
    % 1. Lévy 飞行生成新巢穴
    new_nest = get_cuckoos(nest, bestNest, nVar);
    for i = 1:popSize
        f_new = cs_obj(new_nest(i,:), data, weights);
        if f_new < fitness(i)
            fitness(i) = f_new;
            nest(i,:) = new_nest(i,:);
        end
    end
    
    % 2. 发现并丢弃 (Pa概率)
    new_nest = empty_nests(nest, Pa);
    for i = 1:popSize
        f_new = cs_obj(new_nest(i,:), data, weights);
        if f_new < fitness(i)
            fitness(i) = f_new;
            nest(i,:) = new_nest(i,:);
        end
    end
    
    % 更新全局最优
    [fCurrent, bestIdx] = min(fitness);
    if fCurrent < fMin
        fMin = fCurrent;
        bestNest = nest(bestIdx, :);
    end
    cg_curve(t) = fMin;
end

% ========================== 4. 计算 8 项评价指标 ==========================
MET = toc; 
zbest = bestNest;
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P); 

% 路径数据提取
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
    for k = 1:size(data.FID, 1)
        fixed_distances(k) = data.dis(data.FID(k,1), data.FID(k,2));
    end
end
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

% 输出结果
fprintf('\n==============================================\n');
fprintf('   CS (Cuckoo Search) 性能评价指标\n');
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

% ========================== 辅助函数 ==========================
function new_nest = get_cuckoos(nest, best, dim)
    beta = 1.5;
    sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    for i = 1:size(nest, 1)
        u = randn(1, dim) * sigma;
        v = randn(1, dim);
        step = u ./ abs(v).^(1/beta);
        new_nest(i,:) = nest(i,:) + 0.01 * step .* (nest(i,:) - best);
        new_nest(i,:) = max(min(new_nest(i,:), 1), 0);
    end
end

function new_nest = empty_nests(nest, pa)
    n = size(nest, 1);
    stepsize = rand * (nest(randperm(n),:) - nest(randperm(n),:));
    new_nest = nest + stepsize .* (rand(size(nest)) > pa);
    new_nest = max(min(new_nest, 1), 0);
end

function score = cs_obj(x, S, w)
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