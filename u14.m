%% TOW 疏散分配优化 - 指标集成增强版
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
dim = length(data.DFenPei); 

% 识别固定分配点 (补全固定点数据)
data.FID = []; 
if exist('B','var')
    for k = 1:length(B)
        if length(B{k}) == 1
            data.FID = [data.FID; k, B{k}]; 
        end
    end
end

% ========================== 2. TOW 参数设置 ==========================
popSize = 30;                   
maxGen = 100;                   
weights.w1 = 0.001; weights.w2 = 1.0; 

% 初始化狼群
pop = rand(popSize, dim);
fitness = zeros(popSize, 1);
for i = 1:popSize
    fitness(i) = tow_obj(pop(i,:), data, weights);
end

[~, sortIdx] = sort(fitness);
Alpha_pos = pop(sortIdx(1), :);
Beta_pos = pop(sortIdx(2), :);
fMin = fitness(sortIdx(1));
cg_curve = zeros(1, maxGen);

% ========================== 3. 执行 TOW 进化 ==========================
fprintf('TOW 优化启动，正在计算 8 项评价指标...\n');

for t = 1:maxGen
    a = 2 - t * (2 / maxGen); % 线性收敛因子
    
    for i = 1:popSize
        for j = 1:dim
            % 向 Alpha 狼靠近
            r1 = rand; r2 = rand;
            A1 = 2 * a * r1 - a; C1 = 2 * r2;
            D_alpha = abs(C1 * Alpha_pos(j) - pop(i,j));
            X1 = Alpha_pos(j) - A1 * D_alpha;
            
            % 向 Beta 狼靠近
            r1 = rand; r2 = rand;
            A2 = 2 * a * r1 - a; C2 = 2 * r2;
            D_beta = abs(C2 * Beta_pos(j) - pop(i,j));
            X2 = Beta_pos(j) - A2 * D_beta;
            
            % 更新位置
            pop(i,j) = (X1 + X2) / 2;
        end
    end
    
    % 边界检查与评估
    pop = max(min(pop, 1), 0.01);
    for i = 1:popSize
        fitness(i) = tow_obj(pop(i,:), data, weights);
        if fitness(i) < fMin
            fMin = fitness(i);
            Alpha_pos = pop(i,:);
        elseif fitness(i) > fMin && fitness(i) < fitness(sortIdx(2))
            Beta_pos = pop(i,:);
        end
    end
    cg_curve(t) = fMin;
end

% ========================== 4. 计算 8 项评价指标 ==========================
MET = toc; 
zbest = Alpha_pos;
zbest(zbest < 0.01) = 0.01;
% 动态点分配映射
P = cellfun(@(x) length(x)-1, data.DFenPei);
X_final = ceil(zbest .* P); 

% 汇总所有疏散距离 (动态 + 固定)
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
fprintf('   TOW (Two-Orange Wolves) 性能评价指标\n');
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
function score = tow_obj(x, S, w)
    x(x < 0.01) = 0.01;
    P_tmp = cellfun(@(x) length(x)-1, S.DFenPei);
    X = ceil(x .* P_tmp); 
    total_dist = 0;
    Y = zeros(1, size(S.binan, 1));
    % 累加固定分配
    for k = 1:size(S.FID, 1)
        total_dist = total_dist + S.dis(S.FID(k,1), S.FID(k,2));
        Y(S.FID(k,2)) = Y(S.FID(k,2)) + 12;
    end
    % 累加动态分配
    for i = 1:length(X)
        hID = S.DFenPei{i}(1); eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12; 
    end
    score = w.w1 * total_dist + w.w2 * var(Y); 
end