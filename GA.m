%% GA 疏散分配优化 - 修正版（已删除报错占位符）
clc; clear; close all;

% ========================== 1. 加载与适配数据 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat');
end

% 适配字段：确保 dis 矩阵和 P 向量存在
if exist('dis','var'), data.dis = dis; end
data.P = cellfun(@(x) length(x)-1, data.DFenPei);

% 初始化固定分配负载与距离（针对只有一个避难所选择的点）
data.alldis = 0;
data.YFenPei = zeros(1, size(data.binan, 1));
data.FID = []; 
for k = 1:length(B)
    if length(B{k}) == 1
        targetBinan = B{k};
        data.YFenPei(targetBinan) = data.YFenPei(targetBinan) + 12; 
        data.FID = [data.FID; k, targetBinan];
        data.alldis = data.alldis + data.dis(k, targetBinan);
    end
end

% ========================== 2. GA 参数设置 ==========================
nVar = length(data.DFenPei);  
lb = zeros(1, nVar); ub = ones(1, nVar);           
popSize = 80;       
maxGen = 200;       
pc = 0.85;          
pm = 0.2;           % 保持较高变异率以打破直线收敛

weights.w1 = 0.001; 
weights.w2 = 1.0;   
userData.vMax = 1.2; 

% ========================== 3. 执行 GA 进化 ==========================
fprintf('GA 优化启动，正在搜索最优路径...\n');

% 初始化种群
GApop = rand(popSize, nVar);
fitness = zeros(popSize, 1);
for i = 1:popSize
    fitness(i) = fitness_function(GApop(i,:), data, weights);
end

[bestScore, bestIdx] = min(fitness);
zbest = GApop(bestIdx, :);
cg_curve = zeros(1, maxGen);

for g = 1:maxGen
    % 锦标赛选择：有效解决大数值适应度下的选择压力问题
    newPop = zeros(size(GApop));
    for i = 1:popSize
        candidateIdx = randi(popSize, [1, 3]);
        [~, winner] = min(fitness(candidateIdx));
        newPop(i,:) = GApop(candidateIdx(winner), :);
    end
    
    % 多点交叉
    for i = 1:2:popSize
        if rand < pc
            cp = randi(nVar);
            temp = newPop(i, cp:end);
            newPop(i, cp:end) = newPop(i+1, cp:end);
            newPop(i+1, cp:end) = temp;
        end
    end
    
    % 扰动变异：确保基因变化足以改变离散的出口选择
    for i = 1:popSize
        if rand < pm
            mp = randi(nVar);
            newPop(i, mp) = newPop(i, mp) + 0.2 * randn(); 
            newPop(i, newPop(i,:)>1) = 1;
            newPop(i, newPop(i,:)<0) = 0;
        end
    end
    
    GApop = newPop;
    for i = 1:popSize
        fitness(i) = fitness_function(GApop(i,:), data, weights);
        if fitness(i) < bestScore
            bestScore = fitness(i);
            zbest = GApop(i, :);
        end
    end
    
    % 精英保留：确保最优解不丢失
    GApop(1,:) = zbest;
    fitness(1) = bestScore;
    cg_curve(g) = bestScore;
    
    if mod(g, 20) == 0
        fprintf('代数: %d, 当前最优值: %.2f\n', g, bestScore);
    end
end

% ========================== 4. 结果解码与三图绘制 ==========================
fprintf('===== 优化完成，正在绘制图表 =====\n');

% 解码最优解
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P); 

% 计算疏散时间
numHomes = size(data.start, 1);
evacuationTimes = zeros(numHomes, 1);
for i = 1:size(data.FID, 1)
    hIdx = data.FID(i,1);
    evacuationTimes(hIdx) = (data.dis(hIdx, data.FID(i,2)) * 0.01344) / userData.vMax;
end
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1);
    eIdx = data.DFenPei{i}(X_final(i)+1);
    evacuationTimes(hIdx) = (data.dis(hIdx, eIdx) * 0.01344) / userData.vMax;
end

% --- 图 1: 收敛曲线 ---
figure('Color','w','Name','收敛分析');
plot(cg_curve, 'LineWidth', 2, 'Color', [0.85 0.33 0.1]); 
grid on; title('GA 进化收敛过程'); xlabel('进化代数'); ylabel('适应度值');

% --- 图 2: 地理分配地图 (包含道路底图) ---
figure('Color', 'w', 'Name', '地理分配地图');
hold on;
% 绘制道路底图
if isfield(data, 'road')
    for i = 1:length(data.road)
        road_pts = data.road{i};
        if size(road_pts, 2) >= 2 % 确保有 X, Y 坐标
            plot(road_pts(:,1), road_pts(:,2), 'Color', [0.85 0.85 0.85], 'LineWidth', 0.5); 
        end
    end
end

% 绘制分配路径：GA 优化点用蓝色，固定点用绿色
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1);
    eIdx = data.DFenPei{i}(X_final(i)+1);
    plot([data.start(hIdx,1), data.binan(eIdx,1)], [data.start(hIdx,2), data.binan(eIdx,2)], 'b-', 'LineWidth', 0.5);
end
for i = 1:size(data.FID, 1)
    plot([data.start(data.FID(i,1),1), data.binan(data.FID(i,2),1)], [data.start(data.FID(i,1),2), data.binan(data.FID(i,2),2)], 'g-', 'LineWidth', 0.5);
end

% 绘制标记点
h_bin = scatter(data.binan(:,1), data.binan(:,2), 60, 'g^', 'filled');
h_home = scatter(data.start(:,1), data.start(:,2), 15, 'ro', 'filled');
title('GA 优化：疏散分配地理分布连线图'); 
legend([h_bin, h_home], '避难点', '宅基地');
axis equal; grid on;

% --- 图 3: 疏散时间分布图 ---
figure('Color','w','Name','时间统计');
bar(evacuationTimes, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'none');
hold on; 
line([0, numHomes], [900, 900], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 2);
xlabel('宅基地点编号'); ylabel('疏散时间 (秒)');
title('最终分配方案：各点疏散时间分布 (红色虚线为900s限制)');
grid on;

% ========================== 适应度函数 ==========================
function score = fitness_function(x, S, w)
    x(x < 0.01) = 0.01;
    X = ceil(x .* S.P); 
    total_dist = S.alldis;
    Y = S.YFenPei; 
    for i = 1:length(X)
        hID = S.DFenPei{i}(1); eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12; 
    end
    % 适应度包含总距离 F1 和 负载方差 F2
    score = w.w1 * total_dist + w.w2 * var(Y); 
end