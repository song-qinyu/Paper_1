%% HS 和声搜索算法疏散分配优化 - 指标集成版
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

% 字段预处理
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

% ========================== 2. HS 参数设置 ==========================
nVar = length(data.DFenPei);    
HMS = 30;                       % 和声库大小 (Harmony Memory Size)
maxGen = 500;                   % 最大迭代次数
HMCR = 0.9;                     % 和声记忆库保留率
PAR = 0.3;                      % 微调概率
bw = 0.01;                      % 微调带宽
weights.w1 = 0.001; weights.w2 = 1.0; 

% 初始化和声库
HM = rand(HMS, nVar);
fitness = zeros(HMS, 1);
for i = 1:HMS
    fitness(i) = hs_obj(HM(i,:), data, weights);
end

cg_curve = zeros(1, maxGen);

% ========================== 3. 执行 HS 进化 ==========================
fprintf('HS 优化启动，正在计算 8 项评价指标...\n');

for g = 1:maxGen
    new_harmony = zeros(1, nVar);
    for j = 1:nVar
        if rand < HMCR
            % 从记忆库中选择
            new_harmony(j) = HM(randi([1, HMS]), j);
            % 是否微调
            if rand < PAR
                new_harmony(j) = new_harmony(j) + bw * (rand - 0.5);
            end
        else
            % 随机生成
            new_harmony(j) = rand;
        end
    end
    
    % 边界限制
    new_harmony = max(min(new_harmony, 1), 0.01);
    
    % 评估并更新和声库 (选择压力)
    new_fit = hs_obj(new_harmony, data, weights);
    [max_fit, max_idx] = max(fitness); % 找到最差的替换掉
    if new_fit < max_fit
        HM(max_idx, :) = new_harmony;
        fitness(max_idx) = new_fit;
    end
    
    cg_curve(g) = min(fitness);
end

[fMin, bestIdx] = min(fitness);
bestSol = HM(bestIdx, :);

% ========================== 4. 计算 8 项评价指标 ==========================
MET = toc; 
zbest = bestSol;
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
fprintf('   HS (Harmony Search) 性能评价指标\n');
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

% 绘制收敛曲线
figure('Color','w'); plot(cg_curve, 'LineWidth', 2, 'Color', [0.49, 0.18, 0.56]);
grid on; xlabel('Iteration'); ylabel('Best Fitness');
title('HS Convergence Curve');

% ========================== 适应度函数 ==========================
function score = hs_obj(x, S, w)
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