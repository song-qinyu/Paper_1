clc; clear; close all;
rng(2)
% 自定义放大版的 gaplotpareto 图像窗口
function state = myGaplotPareto(options, state, flag)
    persistent hFig
    if strcmp(flag, 'init')
        hFig = figure('Name', 'NSGA-II Pareto 前沿图（放大）', ...
                      'NumberTitle', 'off', ...
                      'Position', [100, 100, 1000, 700]); 
    elseif strcmp(flag, 'done')
        return;
    end
    figure(hFig);
    plot(state.Score(:,1), state.Score(:,2), 'ro');
    xlabel('目标1：总距离');
    ylabel('目标2：人数方差');
    title('NSGA-II Pareto 前沿图');
    drawnow;
end

% ========== 1) 加载预处理好的数据 ========== 
load('sj5.mat');

% ========== 2) 数据后处理：初始化变量 ========== 
alldis = 0;
FID = [];
YFenPei = zeros(1, size(data.binan, 1));

for i = 1:size(data.start, 1)
    if length(B{i}) <= 1
        YFenPei(B{i}) = YFenPei(B{i}) + 24;
        FID = [FID; i, B{i}];
        alldis = alldis + dis(i, B{i});
    end
end

data.FID = FID;
data.alldis = alldis;
data.YFenPei = YFenPei;
data.dis = dis;

% 初始化路径候选数
P = zeros(1, length(DFenPei));
for i = 1:length(DFenPei)
    P(i) = length(DFenPei{i}) - 1;
end
data.P = P;
data.DFenPei = DFenPei;

% ========== 3) 多目标函数封装 ========== 
function f = evacObjFun_wrapper(x, userData)
    data = userData.S;
    P = data.P;
    dis = data.dis;
    DFenPei = data.DFenPei;
    YFenPei = data.YFenPei;
    alldis = data.alldis;

    x(x < 0.01) = 0.01;
    X = min(ceil(x .* P), P);
    for i = 1:length(X)
        if X(i) >= P(i)
            X(i) = P(i) - 1;
        end
    end

    Y = YFenPei;
    totalTime = 0;

    for i = 1:length(X)
        startID = DFenPei{i}(1);
        binanID = DFenPei{i}(X(i) + 1);
        totalTime = totalTime + dis(startID, binanID);
        Y(binanID) = Y(binanID) + 1;
    end

    f1 = alldis + totalTime;
    f2 = var(Y);
    f = [f1, f2];
end

% ========== 4) NSGA-II 参数设置 ========== 
nVar = length(data.DFenPei);
userData.S = data;

popSize = 100;
maxGen = 100;

lb = zeros(1, nVar);
ub = ones(1,  nVar);

options = optimoptions('gamultiobj', ...
    'PopulationSize', popSize, ...
    'MaxGenerations', maxGen, ...
    'Display', 'iter', ...
    'PlotFcn', {@myGaplotPareto}, ...
    'UseParallel', false, ...
    'UseVectorized', false);

problemFitness = @(x) evacObjFun_wrapper(x, userData);
[xSolutions, fSolutions, exitflag, output, population, score] = ...
    gamultiobj(problemFitness, nVar, [], [], [], [], lb, ub, [], options);

% ========== 5) 后处理计算 Finall 值 ========== 
Finall = [];
for solID = 1:size(xSolutions, 1)
    xVec = xSolutions(solID, :);
    xVec(xVec < 0.01) = 0.01;
    X = min(ceil(xVec .* data.P), data.P);
    for i = 1:length(X)
        if X(i) >= data.P(i)
            X(i) = data.P(i) - 1;
        end
    end

    alldis = data.alldis;
    Y = data.YFenPei;

    for i = 1:length(X)
        alldis = alldis + data.dis(data.DFenPei{i}(1), data.DFenPei{i}(X(i)+1));
        Y(data.DFenPei{i}(X(i)+1)) = Y(data.DFenPei{i}(X(i)+1)) + 1;
    end

    Finall = [Finall; alldis, var(Y)];
end

% ========== 6) 结果排序与展示 ========== 
[~, idx] = sort(Finall(:, 2));
xSolutions = xSolutions(idx, :);
FF = Finall(idx, :);

solID = find(Finall(:, 2) == min(Finall(:, 2)));
xVec = xSolutions(solID, :);
xVec(xVec < 0.01) = 0.01;
X = min(ceil(xVec .* data.P), data.P);
for i = 1:length(X)
    if X(i) >= data.P(i)
        X(i) = data.P(i) - 1;
    end
end

Y = data.YFenPei;
alldis = data.alldis;
for i = 1:length(X)
    alldis = alldis + data.dis(data.DFenPei{i}(1), data.DFenPei{i}(X(i)+1));
    Y(data.DFenPei{i}(X(i)+1)) = Y(data.DFenPei{i}(X(i)+1)) + 1;
end

% 打印避难点人数
fprintf('最均衡方案下每个避难点的人数为：\n');
for i = 1:length(Y)
    fprintf('避难点 %d: %d 人\n', i, Y(i));
end

% 绘制人数柱状图
figure;
bar(Y);
xlabel('避难点编号'); ylabel('对应人数');
title('每个避难点对应的人数（最均衡方案）');
set(gca, 'FontName', 'Microsoft YaHei');
print(gcf, '-djpeg', '-r300', '避难点人数分布_最均衡方案.jpg');

% ========== 7) 绘制路径图并标注避难点编号 ========== 
figure;
for i = 1:length(data.road)
    temp = data.road{i};
    plot(temp(:,1), temp(:,2), 'k-'); hold on;
end
for i = 1:size(data.FID, 1)
    a = [data.start(data.FID(i,1),1), data.binan(data.FID(i,2),1)];
    b = [data.start(data.FID(i,1),2), data.binan(data.FID(i,2),2)];
    plot(a, b, 'b-'); hold on;
end
for i = 1:length(X)
    a = [data.start(data.DFenPei{i}(1),1), data.binan(data.DFenPei{i}(X(i)+1),1)];
    b = [data.start(data.DFenPei{i}(1),2), data.binan(data.DFenPei{i}(X(i)+1),2)];
    plot(a, b, 'b-'); hold on;
end
h1 = scatter(data.binan(:,1), data.binan(:,2), 18, 'g^', 'filled');
h2 = scatter(data.start(:,1), data.start(:,2), 8, 'ro', 'filled');

% ✅ 添加避难点编号（与人数柱状图对应）
for i = 1:size(data.binan, 1)
    text(data.binan(i, 1), data.binan(i, 2), [' ' num2str(i)], ...
         'Color', 'g', 'FontSize', 8, 'VerticalAlignment', 'top');
end

title('每个疏散点疏散人数最均衡（总拥挤度）方案（避难点编号已标注）');
legend([h1, h2], '避难点', '宅基地');
set(gca, 'FontName', 'Microsoft YaHei');
print(gcf, '-djpeg', '-r300', '最均衡方案路径图_编号标注.jpg');
