%% SSA 疏散分配优化运行脚本
clc; clear; close all;

% ========================== 1. 加载与适配数据 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat');
end

% 适配字段
if exist('dis','var'), data.dis = dis; end
% P 代表每个待优化宅基地对应的可选避难所数量
data.P = cellfun(@(x) length(x)-1, data.DFenPei);

% 初始化固定分配负载与距离 (FID)
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

% ========================== 2. 算法参数设置 ==========================
pop = 50;           % 种群数量
M = 100;            % 最大迭代次数
dim = length(data.DFenPei); % 变量维度 (待优化宅基地数量)
c = 0.01;           % 变量下界 (lb)
d = 1.0;            % 变量上界 (ub)

% 定义目标函数句柄
fobj = @(x) fitness_function(x, data);

% 地图栅格数据 G (用于 SSA 内部初始化)
% 假设 G 已经在 sj5.mat 中，如果不存在，请确保定义它
if ~exist('G', 'var')
    G = zeros(100, dim + 1); % 示例占位，请根据实际地图定义
end

% ========================== 3. 执行 SSA 优化 ==========================
fprintf('SSA 算法启动...\n');
% 调用上传的 SSA 函数
% [fMin, bestX, Convergence_curve] = SSA(pop, M, lb, ub, dim, fobj, G)
[fMin, bestX, Convergence_curve] = SSA(pop, M, c, d, dim, fobj, G); %

% ========================== 4. 结果可视化与解码 ==========================
% 解码最优位置
bestX(bestX < 0.01) = 0.01;
X_final = ceil(bestX .* data.P); 

% 计算疏散时间
numHomes = size(data.start, 1);
evacuationTimes = zeros(numHomes, 1);
vMax = 1.2; 

% 统计所有连接 (固定 + 优化)
for i = 1:size(data.FID, 1)
    hIdx = data.FID(i,1);
    evacuationTimes(hIdx) = (data.dis(hIdx, data.FID(i,2)) * 0.01344) / vMax;
end
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1);
    eIdx = data.DFenPei{i}(X_final(i)+1);
    evacuationTimes(hIdx) = (data.dis(hIdx, eIdx) * 0.01344) / vMax;
end

% 绘制收敛曲线
figure('Color','w');
plot(Convergence_curve, 'LineWidth', 2, 'Color', 'r');
grid on; title('SSA 进化收敛过程'); xlabel('迭代次数'); ylabel('适应度值');

% 绘制全量地理地图
figure('Color', 'w'); hold on;
if isfield(data, 'road'), for i=1:length(data.road), plot(data.road{i}(:,1), data.road{i}(:,2), 'Color', [0.8 0.8 0.8]); end; end
for i = 1:size(data.FID, 1)
    plot([data.start(data.FID(i,1),1), data.binan(data.FID(i,2),1)], [data.start(data.FID(i,1),2), data.binan(data.FID(i,2),2)], 'g-');
end
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1); eIdx = data.DFenPei{i}(X_final(i)+1);
    plot([data.start(hIdx,1), data.binan(eIdx,1)], [data.start(hIdx,2), data.binan(eIdx,2)], 'b-');
end
scatter(data.binan(:,1), data.binan(:,2), 60, 'g^', 'filled');
scatter(data.start(:,1), data.start(:,2), 20, 'ro', 'filled');
title('SSA 优化：全量疏散分配地图'); axis equal; grid on;

% ========================== 适应度函数 ==========================
function score = fitness_function(x, S)
    w1 = 0.001; w2 = 1.0;
    x(x < 0.01) = 0.01;
    X = ceil(x .* S.P); 
    total_dist = S.alldis;
    Y = S.YFenPei; 
    for i = 1:length(X)
        hID = S.DFenPei{i}(1); eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12; 
    end
    score = w1 * total_dist + w2 * var(Y); 
end