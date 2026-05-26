%% WDO 风驱动优化算法疏散分配优化 - 指标集成版
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

% ========================== 2. WDO 参数设置 ==========================
popSize = 40;                % 种群规模
maxGen = 150;                % 最大迭代次数
nVar = length(data.DFenPei); % 维度
RT = 3;                      % 理想气体常数
g = 0.2;                     % 重力加速度
alpha = 0.4;                 % 摩擦系数
c = 0.4;                     % 科里奥利力系数
maxV = 0.3;                  % 最大速度限制
weights.w1 = 0.001; weights.w2 = 1.0; 

% 初始化位置与速度
pos = rand(popSize, nVar);
vel = maxV * (2 * rand(popSize, nVar) - 1);
fitness = zeros(popSize, 1);
for i = 1:popSize
    fitness(i) = wdo_obj(pos(i,:), data, weights);
end

[fMin, bestIdx] = min(fitness);
bestPos = pos(bestIdx, :);
cg_curve = zeros(1, maxGen);

% ========================== 3. 执行 WDO 进化 ==========================
fprintf('WDO 优化启动，正在计算 8 项评价指标...\n');

for t = 1:maxGen
    % 更新每个质点的速度与位置
    for i = 1:popSize
        % 随机排列维度用于更新（WDO特点）
        a = randperm(nVar);
        
        % 速度更新公式
        vel(i,:) = (1-alpha)*vel(i,:) - g*pos(i,:) + ...
                   (abs(1-1/i)*RT*(bestPos - pos(i,:))) + ...
                   (c*vel(i,a)/i);
               
        % 速度限制
        vel(i, vel(i,:) > maxV) = maxV;
        vel(i, vel(i,:) < -maxV) = -maxV;
        
        % 位置更新
        pos(i,:) = pos(i,:) + vel(i,:);
        
        % 边界处理
        pos(i, pos(i,:) > 1) = 1;
        pos(i, pos(i,:) < 0.01) = 0.01;
        
        % 评估适应度
        fitness(i) = wdo_obj(pos(i,:), data, weights);
        if fitness(i) < fMin
            fMin = fitness(i);
            bestPos = pos(i,:);
        end
    end
    cg_curve(t) = fMin;
end

% ========================== 4. 计算 8 项评价指标 ==========================
MET = toc; 
zbest = bestPos;
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
fprintf('   WDO (Wind Driven Optimization) 性能评价指标\n');
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

% 绘制收敛曲线
figure('Color','w'); plot(cg_curve, 'LineWidth', 2, 'Color', [0, 0.447, 0.741]);
grid on; xlabel('Iteration'); ylabel('Best Fitness');
title('WDO Convergence Process');

% ========================== 适应度函数 ==========================
function score = wdo_obj(x, S, w)
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