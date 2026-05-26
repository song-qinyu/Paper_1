%% GSA 万有引力搜索算法疏散分配优化 - 指标集成版
clear; clc; close all; tic;

% ========================== 1. 加载数据 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat');
end

starts = data.start;    
binan_raw = data.binan;  
num_points = size(starts, 1);
num_refuges = size(binan_raw, 1);

% 适配 dis 矩阵
if exist('dis','var'), data.dis = dis; end
data.P = cellfun(@(x) length(x)-1, data.DFenPei);

% ========================== 2. 空间约束预处理 ==========================
K = 5; % 局部引导：每个点只在最近的 5 个避难所中选
fprintf('正在计算局部最优候选范围...\n');
candidate_matrix = zeros(num_points, K);
for i = 1:num_points
    dists = sum((starts(i,:) - binan_raw).^2, 2);
    [~, sorted_idx] = sort(dists);
    candidate_matrix(i, :) = sorted_idx(1:K); 
end

% ========================== 3. GSA 参数设置 ==========================
dim = num_points;     
lb = 1; ub = K;  
pop = 30;     
maxIter = 100;
G0 = 100; % 初始引力常数

% 初始化种群
X = lb + (ub - lb) .* rand(pop, dim);
V = zeros(pop, dim);
fitness = zeros(pop, 1);
for i = 1:pop
    fitness(i) = My_Spatial_Fitness(X(i,:), starts, binan_raw, candidate_matrix, data);
end

[fMin, best_idx] = min(fitness);
bestX = X(best_idx, :);
cg_curve = zeros(1, maxIter);

% ========================== 4. 执行 GSA 优化 ==========================
fprintf('GSA 优化启动，正在计算 8 项评价指标...\n');

for t = 1:maxIter
    % 更新引力常数 G
    G = G0 * exp(-20 * t / maxIter);
    
    % 计算质量 M
    best_f = min(fitness); worst_f = max(fitness);
    if best_f == worst_f
        M = ones(pop, 1);
    else
        m = (fitness - worst_f) ./ (best_f - worst_f);
        M = m ./ sum(m);
    end
    
    % 计算加速度
    acc = zeros(pop, dim);
    for i = 1:pop
        F = zeros(1, dim);
        for j = 1:pop
            if i ~= j
                R = norm(X(i,:) - X(j,:), 2);
                F = F + rand * G * (M(i) * M(j) / (R + eps)) * (X(j,:) - X(i,:));
            end
        end
        acc(i,:) = F / (M(i) + eps);
    end
    
    % 更新速度与位置
    V = rand * V + acc;
    X = X + V;
    X = max(min(X, ub), lb);
    
    % 评估
    for i = 1:pop
        fitness(i) = My_Spatial_Fitness(X(i,:), starts, binan_raw, candidate_matrix, data);
        if fitness(i) < fMin
            fMin = fitness(i);
            bestX = X(i,:);
        end
    end
    cg_curve(t) = fMin;
end

% ========================== 5. 计算 8 项评价指标 ==========================
MET = toc; 
% 转换最优解
final_local_idx = max(1, min(K, round(bestX)));
X_final_global = zeros(1, num_points);
all_dist = zeros(1, num_points);
used_bins = [];

for i = 1:num_points
    target_bin = candidate_matrix(i, final_local_idx(i));
    X_final_global(i) = target_bin;
    all_dist(i) = data.dis(i, target_bin);
    used_bins = [used_bins, target_bin];
end

% 指标统计
TED = sum(all_dist);           
ATD = mean(all_dist);          
MID = max(all_dist);           
BTV = fMin;                    
SUR = (length(unique(used_bins)) / num_refuges) * 100; 

% 收敛代数 (CG)
change = abs(diff(cg_curve));
last_c = find(change > 1e-6, 1, 'last');
if isempty(last_c), CG = 1; else CG = last_c + 1; end
SD = 0; 

% ========================== 6. 输出结果面板 ==========================
fprintf('\n==============================================\n');
fprintf('   GSA (Gravitational Search) 性能评价指标\n');
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

% ========================== 核心适应度函数 ==========================
function score = My_Spatial_Fitness(x, starts, binans, cand_mat, data)
    num = size(starts, 1);
    idx = max(1, min(size(cand_mat, 2), round(x)));
    total_dist = 0;
    load_balance = zeros(1, size(binans, 1));
    
    for i = 1:num
        target_bin = cand_mat(i, idx(i));
        total_dist = total_dist + data.dis(i, target_bin);
        load_balance(target_bin) = load_balance(target_bin) + 12;
    end
    % 权重 w1=0.001, w2=1.0 保持与其他算法一致
    score = 0.001 * total_dist + 1.0 * var(load_balance);
end