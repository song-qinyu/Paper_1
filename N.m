clc; clear; close all;
rng(2)

% 加载数据
load('sj5.mat');

%% 1. 数据预处理
alldis = 0;
FID = [];
YFenPei = zeros(1, size(data.binan, 1)); 
for i = 1:size(data.start, 1)
    if length(B{i}) <= 1
        YFenPei(B{i}) = YFenPei(B{i}) + 12;
        FID = [FID; i, B{i}];
        alldis = alldis + dis(i, B{i});
    end
end
data.FID = FID;
data.alldis = alldis;
P = [];
for i = 1:length(DFenPei)
    P(i) = length(DFenPei{i}) - 1;
end
data.P = P;
data.dis = dis;
nVar = length(data.DFenPei); 

% 参数设置
userData.S = data;

% --- 颜色定义 ---
springGreen   = [0.44, 0.78, 0.45];   % 春日青 (避难所)
deepBlue      = [0.00, 0.35, 0.65];   % 深蓝色 (宅基地)
lineLightBlue = [0.65, 0.80, 0.95];   % 浅蓝色 (均衡方案线)
conflictRed   = [1.00, 0.20, 0.20];   % 鲜红色 (不一致分配线)

%% 2. 运行 NSGA-II
popSize = 100;   
maxGen  = 100;   
options = optimoptions('gamultiobj', 'PopulationSize', popSize, 'MaxGenerations', maxGen, 'Display', 'iter');

problemFitness = @(x) evacObjFun_wrapper(x, userData);
[xSolutions, fSolutions, ~, ~] = gamultiobj(problemFitness, nVar, [],[],[],[], zeros(1,nVar), ones(1,nVar), [], options);

% 计算最终指标
Finall = [];
for solID = 1:size(xSolutions, 1)
    xVec = xSolutions(solID,:); xVec(xVec < 0.01) = 0.01;
    X = ceil(xVec .* data.P);
    currDis = data.alldis; currY = data.YFenPei;
    for i = 1:length(X)
        currDis = currDis + data.dis(data.DFenPei{i}(1), data.DFenPei{i}(X(i)+1));
        currY(data.DFenPei{i}(X(i)+1)) = currY(data.DFenPei{i}(X(i)+1)) + 12;
    end
    Finall = [Finall; currDis, var(currY)];
end

%% 3. 方案比对逻辑
solID_Eq = find(Finall(:, 2) == min(Finall(:, 2)), 1);
X_Eq = ceil(max(xSolutions(solID_Eq,:), 0.01) .* data.P);

solID_Time = find(Finall(:, 1) == min(Finall(:, 1)), 1);
X_Time = ceil(max(xSolutions(solID_Time,:), 0.01) .* data.P);

%% 4. 绘制超大范围对比图
% 设置超宽画布 [左 下 宽 高]
figure('Color', 'w', 'Position', [50, 50, 1400, 800]); 
hold on;

% A. 绘制“最均衡”方案线 (浅蓝色)
for i = 1:size(data.FID, 1)
    plot([data.start(data.FID(i, 1), 1), data.binan(data.FID(i, 2), 1)], ...
         [data.start(data.FID(i, 1), 2), data.binan(data.FID(i, 2), 2)], ...
         'Color', lineLightBlue, 'LineWidth', 1); 
end
for i = 1:length(X_Eq)
    plot([data.start(data.DFenPei{i}(1), 1), data.binan(data.DFenPei{i}(X_Eq(i)+1), 1)], ...
         [data.start(data.DFenPei{i}(1), 2), data.binan(data.DFenPei{i}(X_Eq(i)+1), 2)], ...
         'Color', lineLightBlue, 'LineWidth', 1);
end

% B. 叠加“不一致”的线 (红色)
conflictCount = 0;
for i = 1:length(X_Eq)
    if X_Eq(i) ~= X_Time(i)
        plot([data.start(data.DFenPei{i}(1), 1), data.binan(data.DFenPei{i}(X_Time(i)+1), 1)], ...
             [data.start(data.DFenPei{i}(1), 2), data.binan(data.DFenPei{i}(X_Time(i)+1), 2)], ...
             'Color', conflictRed, 'LineWidth', 1.5);
        conflictCount = conflictCount + 1;
    end
end

% C. 绘制点位
h2 = scatter(data.start(:,1), data.start(:,2), 22, deepBlue, 'o', 'filled'); 
h1 = scatter(data.binan(:,1), data.binan(:,2), 70, springGreen, '^', 'filled', 'MarkerEdgeColor', [0.2 0.2 0.2]); 

% D. 极大幅度调整坐标轴
allX = [data.start(:,1); data.binan(:,1)]; 
allY = [data.start(:,2); data.binan(:,2)];
dx = max(allX) - min(allX); 
dy = max(allY) - min(allY);

% 核心修改：留白比例增加到 0.3，让轴显得更长
margin = 0.30; 
xlim([min(allX) - dx*margin, max(allX) + dx*margin]);
ylim([min(allY) - dy*margin, max(allY) + dy*margin]);

title(['方案对比图 (检测到 ', num2str(conflictCount), ' 个分配冲突点)'], 'FontSize', 15);
legend([h1, h2], '避难点', '宅基地', 'Location', 'northeastoutside');
xlabel('X 坐标 (meters)'); ylabel('Y 坐标 (meters)');
grid on; box on;
set(gca, 'FontName', 'Microsoft YaHei', 'FontSize', 11, 'TickDir', 'out');

% 保存
saveas(gcf, '方案对比冲突图_大坐标轴.jpg');
fprintf('生成完毕！坐标轴范围已大幅扩充，留白比例为 30%%。\n');

%% --- 目标函数 ---
function f = evacObjFun_wrapper(x, userData)
    S = userData.S;
    x(x < 0.01) = 0.01;
    X = ceil(x .* S.P);
    alldis = S.alldis; Y = S.YFenPei;
    for i = 1:length(X)
        alldis = alldis + S.dis(S.DFenPei{i}(1), S.DFenPei{i}(X(i)+1));
        Y(S.DFenPei{i}(X(i)+1)) = Y(S.DFenPei{i}(X(i)+1)) + 12;
    end
    f = [alldis, var(Y)];
end