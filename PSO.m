%% PSO 疏散分配优化 - 最终修正整合版
clc; clear; close all;

% ========================== 1. 加载数据 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat');
end

% ========================== 2. 数据适配与预处理 ==========================
fprintf('正在适配数据字段...\n');

% 将独立的 dis 变量放入 data 结构体中
if exist('dis','var')
    data.dis = dis;
end

% 计算 P (每个待分配点的可选出口数量)
if isfield(data, 'DFenPei')
    data.P = cellfun(@(x) length(x)-1, data.DFenPei);
else
    error('数据中缺失 DFenPei 字段');
end

% 重新计算已固定分配点的初始总距离和初始负载
data.alldis = 0;
data.YFenPei = zeros(1, size(data.binan, 1));
data.FID = []; % 记录固定分配的 ID

for k = 1:length(B)
    if length(B{k}) == 1
        targetBinan = B{k};
        data.YFenPei(targetBinan) = data.YFenPei(targetBinan) + 12; 
        data.FID = [data.FID; k, targetBinan];
        data.alldis = data.alldis + data.dis(k, targetBinan);
    end
end

% ========================== 3. PSO 核心初始化 ==========================
nVar = length(data.DFenPei);  
lb = zeros(1, nVar); 
ub = ones(1, nVar);           
N = 50;           % 粒子群规模
maxIter = 100;    % 迭代次数
w_pso = 0.8; 
c1 = 1.2; 
c2 = 1.2;         

weights.w1 = 0.001; % 距离权重
weights.w2 = 1.0;   % 拥挤度权重
userData.vMax = 1.2; 

vel = zeros(N, nVar);
pos = repmat(lb, N, 1) + rand(N, nVar) .* repmat((ub - lb), N, 1);
pBest = pos; 
pBestScore = inf(N, 1);
gBest = zeros(1, nVar); 
gBestScore = inf;
cg_curve = zeros(1, maxIter);
Vmax = 0.15 * (ub - lb); 

% ========================== 4. 执行 PSO 迭代 ==========================
fprintf('数据适配完成。PSO 优化开始迭代...\n');

for l = 1:maxIter
    for i = 1:N
        % 计算适应度
        fitness = fitness_function(pos(i,:), data, weights);
        
        % 更新个体最优
        if fitness < pBestScore(i)
            pBestScore(i) = fitness;
            pBest(i,:) = pos(i,:);
        end
        
        % 更新全局最优
        if fitness < gBestScore
            gBestScore = fitness;
            gBest = pos(i,:);
        end
    end
    
    % 更新位置与速度
    for i = 1:N
        vel(i,:) = w_pso*vel(i,:) + c1*rand()*(pBest(i,:) - pos(i,:)) + c2*rand()*(gBest - pos(i,:));
        
        % 边界限制
        vel(i, vel(i,:) > Vmax) = Vmax(vel(i,:) > Vmax);
        vel(i, vel(i,:) < -Vmax) = -Vmax(vel(i,:) < -Vmax);
        
        pos(i,:) = pos(i,:) + vel(i,:);
        
        pos(i, pos(i,:) > ub) = ub(pos(i,:) > ub);
        pos(i, pos(i,:) < lb) = lb(pos(i,:) < lb);
    end
    cg_curve(l) = gBestScore;
    
    if mod(l, 10) == 0
        fprintf('迭代进度: %d%%, 当前最优得分: %.2f\n', (l/maxIter)*100, gBestScore);
    end
end

% ========================== 5. 结果解码与可视化 ==========================
fprintf('===== 优化完成，正在生成图表 =====\n');

% 解码全局最优解
gBest(gBest < 0.01) = 0.01;
X_final = ceil(gBest .* data.P); 

% 计算最终所有点的疏散时间
numHomes = size(data.start, 1);
evacuationTimes = zeros(numHomes, 1);
for i = 1:size(data.FID, 1)
    evacuationTimes(data.FID(i,1)) = (data.dis(data.FID(i,1), data.FID(i,2)) * 0.01344) / userData.vMax;
end
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1);
    eIdx = data.DFenPei{i}(X_final(i)+1);
    evacuationTimes(hIdx) = (data.dis(hIdx, eIdx) * 0.01344) / userData.vMax;
end

% --- 图表 1: 收敛曲线 ---
figure('Color','w','Name','PSO收敛曲线');
plot(cg_curve, 'LineWidth', 2, 'Color', [0.85 0.33 0.1]); 
grid on; title('PSO 综合适应度收敛过程'); xlabel('迭代代数'); ylabel('加权目标函数值');

% --- 图表 2: 地理分配连线图 ---
figure('Color', 'w', 'Name', 'PSO最终分配方案图');
hold on;
% 绘制道路背景
for i = 1:length(data.road)
    plot(data.road{i}(:,1), data.road{i}(:,2), 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5); 
end
% 绘制分配路径 (蓝色表示 PSO 优化路径)
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1);
    eIdx = data.DFenPei{i}(X_final(i)+1);
    plot([data.start(hIdx,1), data.binan(eIdx,1)], ...
         [data.start(hIdx,2), data.binan(eIdx,2)], 'b-', 'LineWidth', 0.5);
end
% 绘制固定路径 (绿色)
for i = 1:size(data.FID, 1)
    plot([data.start(data.FID(i,1),1), data.binan(data.FID(i,2),1)], ...
         [data.start(data.FID(i,1),2), data.binan(data.FID(i,2),2)], 'g-', 'LineWidth', 0.5);
end
h_binan = scatter(data.binan(:,1), data.binan(:,2), 50, 'g^', 'filled');
h_start = scatter(data.start(:,1), data.start(:,2), 15, 'ro', 'filled');
title(['PSO 疏散方案地理分布 (得分: ', num2str(gBestScore), ')']);
legend([h_binan, h_start], '避难点', '宅基地');
axis equal; grid on;

% --- 图表 3: 疏散时间分布图 ---
figure('Color','w','Name','疏散时间分布');
bar(evacuationTimes, 'EdgeColor', 'none', 'FaceColor', [0.2 0.4 0.7]);
hold on;
line([0, numHomes], [900, 900], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5); % 900s 警戒线
xlabel('宅基地点编号'); ylabel('疏散时间 (秒)');
title('各宅基地点最终疏散时间');
xticks(1:500:numHomes); grid on;

fprintf('最终最优得分: %.4f\n', gBestScore);
fprintf('平均疏散时间: %.2f 秒\n', mean(evacuationTimes));

% ========================== 6. 适应度函数 ==========================
function [score, F1, F2] = fitness_function(x, S, w)
    x(x < 0.01) = 0.01;
    X = ceil(x .* S.P); 
    total_dist = S.alldis;
    Y = S.YFenPei; 
    for i = 1:length(X)
        hID = S.DFenPei{i}(1);
        eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12; 
    end
    F1 = total_dist;
    F2 = var(Y); 
    score = w.w1 * F1 + w.w2 * F2; 
end