clc; clear; close all;
rng(2);

%% ========== 1) 加载预处理数据 ========== 
load('sj5.mat');

%% ========== 2) 数据后处理：初始化变量 ==========
alldis = 0;
FID = [];
YFenPei = zeros(1, size(data.binan, 1));

for i = 1:size(data.start, 1)
    if length(B{i}) <= 1
        YFenPei(B{i}) = YFenPei(B{i}) + 24;  %%一个宅基地的人数是24人
        FID = [FID; i, B{i}];
        alldis = alldis + dis(i, B{i});
    end
end

% 更新 data 结构体
data.FID = FID;
data.alldis = alldis;
data.YFenPei = YFenPei;
data.dis = dis;

% 初始化路径候选数 P
P = zeros(1, length(DFenPei));
for i = 1:length(DFenPei)
    P(i) = length(DFenPei{i}) - 1;
end
data.P = P;

%% ========== 3) 多目标优化函数封装 ==========
function f = evacObjFun_wrapper(x, userData)
    data = userData.S;
    P = data.P;
    dis = data.dis;
    DFenPei = data.DFenPei;
    YFenPei = data.YFenPei;
    alldis = data.alldis;

    x(x < 0.01) = 0.01;
    X = min(ceil(x .* P), P);  % 保证不越界

    Y = YFenPei;
    totalTime = 0;

    for i = 1:length(X)
        if X(i)+1 <= length(DFenPei{i})
            startID = DFenPei{i}(1);
            binanID = DFenPei{i}(X(i) + 1);
            totalTime = totalTime + dis(startID, binanID);
            Y(binanID) = Y(binanID) + 1;
        end
    end

    f1 = alldis + totalTime;   % 距离总和
    f2 = var(Y);               % 人数方差
    f = [f1, f2];
end

%% ========== 4) NSGA-II 参数设置 ==========
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
    'PlotFcn', {@gaplotpareto}, ...
    'UseParallel', false, ...
    'UseVectorized', false);

problemFitness = @(x) evacObjFun_wrapper(x, userData);

[xSolutions, fSolutions] = gamultiobj(problemFitness, nVar, [], [], [], [], lb, ub, [], options);

%% ========== 5) 解的后处理计算 ==========
Finall = [];
for solID = 1:size(xSolutions, 1)
    xVec = xSolutions(solID, :);
    xVec(xVec < 0.01) = 0.01;
    X = min(ceil(xVec .* data.P), data.P);

    alldis = data.alldis;
    Y = data.YFenPei;

    for i = 1:length(X)
        if X(i)+1 <= length(data.DFenPei{i})
            startID = data.DFenPei{i}(1);
            binanID = data.DFenPei{i}(X(i) + 1);
            alldis = alldis + data.dis(startID, binanID);
            Y(binanID) = Y(binanID) + 1;
        end
    end

    Finall = [Finall; [alldis, var(Y)]];
end

%% ========== 6) 结果排序与展示 ==========
[~, idx] = sort(Finall(:, 2));  % 按人数方差升序排序
xSolutions = xSolutions(idx, :);
FF = Finall(idx, :);

% 取最均衡方案
xVec = xSolutions(1, :);
xVec(xVec < 0.01) = 0.01;
X = min(ceil(xVec .* data.P), data.P);

Y = data.YFenPei;
alldis = data.alldis;
for i = 1:length(X)
    if X(i)+1 <= length(data.DFenPei{i})
        binanID = data.DFenPei{i}(X(i)+1);
        Y(binanID) = Y(binanID) + 1;
        alldis = alldis + data.dis(data.DFenPei{i}(1), binanID);
    end
end

% 打印结果
fprintf('最均衡方案下每个避难点的人数为：\n');
for i = 1:length(Y)
    fprintf('避难点 %d: %d 人\n', i, Y(i));
end

% 绘制人数分布柱状图
figure;
bar(Y);
xlabel('避难点编号'); ylabel('对应人数');
title('每个避难点对应的人数（最均衡方案）');
set(gca, 'FontName', 'Microsoft YaHei');
print(gcf, '-djpeg', '-r300', '避难点人数分布_最均衡方案.jpg');

% 绘制路径连接图
figure;
hold on;
for i = 1:length(data.road)
    temp = data.road{i};
    plot(temp(:,1), temp(:,2), 'k-');
end
for i = 1:size(data.FID, 1)
    a = [data.start(data.FID(i,1),1), data.binan(data.FID(i,2),1)];
    b = [data.start(data.FID(i,1),2), data.binan(data.FID(i,2),2)];
    plot(a, b, 'b-');
end
for i = 1:length(X)
    if X(i)+1 <= length(data.DFenPei{i})
        a = [data.start(data.DFenPei{i}(1),1), data.binan(data.DFenPei{i}(X(i)+1),1)];
        b = [data.start(data.DFenPei{i}(1),2), data.binan(data.DFenPei{i}(X(i)+1),2)];
        plot(a, b, 'b-');
    end
end
h1 = scatter(data.binan(:,1), data.binan(:,2), 18, 'g^', 'filled');
h2 = scatter(data.start(:,1), data.start(:,2), 8, 'ro', 'filled');
title('每个疏散点疏散人数最均衡（总拥挤度）方案');
legend([h1, h2], '避难点', '宅基地');
set(gca, 'FontName', 'Microsoft YaHei');
print(gcf, '-djpeg', '-r300', '每个疏散点疏散人数最均衡（总拥挤度）方案.jpg');
