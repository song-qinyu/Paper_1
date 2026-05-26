%% GWO 疏散分配优化集成主程序
clc; clear; close all;

% ========================== 1. 加载与预处理数据 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat，请确保该文件在当前文件夹下');
end

% 适配字段与预处理 (保持与你原始逻辑一致)
if exist('dis','var'), data.dis = dis; end
data.P = cellfun(@(x) length(x)-1, data.DFenPei);
data.YFenPei = zeros(1, size(data.binan, 1));
data.alldis = 0;
data.FID = []; 
optimized_home_ids = cellfun(@(x) x(1), data.DFenPei);

for k = 1:size(data.start, 1)
    if ~ismember(k, optimized_home_ids)
        targetBinan = B{k}(1);
        data.YFenPei(targetBinan) = data.YFenPei(targetBinan) + 12; 
        data.FID = [data.FID; k, targetBinan];
        data.alldis = data.alldis + data.dis(k, targetBinan);
    end
end

% ========================== 2. GWO 算法参数设置 ==========================
SearchAgents_no = 50;  % 种群规模
Max_iter = 100;        % 最大迭代次数
dim = length(data.DFenPei); 
lb = 0.01; 
ub = 1.0;

% ========================== 3. 调用 GWO 核心寻优 ==========================
fprintf('GWO 算法启动，正在优化 %d 个点的分配方案...\n', dim);

% 定义适应度函数句柄
fobj = @(x) internal_fitness(x, data);

% 执行 GWO
[Best_score, Best_pos, Convergence_curve] = GWO_Core(SearchAgents_no, Max_iter, lb, ub, dim, fobj);

% ========================== 4. 结果可视化 (3张图) ==========================
fprintf('优化完成，正在生成分析图表...\n');
X_final = ceil(Best_pos .* data.P); 

% 计算时间数据
vMax = 1.2; 
numTotal = size(data.start, 1);
evacuationTimes = zeros(numTotal, 1);

% 【图 1: 地理分配地图】
figure('Color', 'w', 'Name', 'GWO 地理分配地图'); hold on;
if isfield(data, 'road')
    for i = 1:length(data.road), plot(data.road{i}(:,1), data.road{i}(:,2), 'Color', [0.9 0.9 0.9]); end
end

for i = 1:size(data.FID, 1)
    hIdx = data.FID(i,1); eIdx = data.FID(i,2);
    plot([data.start(hIdx,1), data.binan(eIdx,1)], [data.start(hIdx,2), data.binan(eIdx,2)], 'g-', 'LineWidth', 0.5);
    evacuationTimes(hIdx) = (data.dis(hIdx, eIdx) * 0.01344) / vMax;
end
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1); eIdx = data.DFenPei{i}(X_final(i)+1);
    plot([data.start(hIdx,1), data.binan(eIdx,1)], [data.start(hIdx,2), data.binan(eIdx,2)], 'b-', 'LineWidth', 0.5);
    evacuationTimes(hIdx) = (data.dis(hIdx, eIdx) * 0.01344) / vMax;
end
scatter(data.binan(:,1), data.binan(:,2), 80, 'g^', 'filled');
scatter(data.start(:,1), data.start(:,2), 20, 'ro', 'filled');
title('GWO 疏散分配方案'); axis equal; grid on;

% 【图 2: 收敛曲线】
figure('Color','w'); plot(Convergence_curve, 'r', 'LineWidth', 2);
xlabel('迭代次数'); ylabel('综合适应度值'); title('GWO 算法收敛过程');

% 【图 3: 疏散时间分布】
figure('Color','w'); bar(evacuationTimes, 'FaceColor', [0.2 0.6 0.8]);
hold on; line([0, numTotal], [900, 900], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
xlabel('宅基地编号'); ylabel('疏散时间 (秒)'); title('GWO 疏散时间分布图');

% ========================== 核心函数库 ==========================

function [Alpha_score, Alpha_pos, Convergence_curve] = GWO_Core(SearchAgents_no, Max_iter, lb_val, ub_val, dim, fobj)
    Alpha_pos = zeros(1, dim); Alpha_score = inf;
    Beta_pos = zeros(1, dim); Beta_score = inf;
    Delta_pos = zeros(1, dim); Delta_score = inf;
    
    % 初始化 (修正为连续概率空间)
    Positions = lb_val + (ub_val - lb_val) .* rand(SearchAgents_no, dim);
    Convergence_curve = zeros(1, Max_iter);
    
    for l = 1:Max_iter
        for i = 1:size(Positions, 1)
            % 边界处理
            Positions(i,:) = max(min(Positions(i,:), ub_val), lb_val);
            fitness = fobj(Positions(i,:));
            
            % 更新 Alpha, Beta, Delta
            if fitness < Alpha_score 
                Alpha_score = fitness; Alpha_pos = Positions(i,:);
            elseif fitness > Alpha_score && fitness < Beta_score 
                Beta_score = fitness; Beta_pos = Positions(i,:);
            elseif fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score 
                Delta_score = fitness; Delta_pos = Positions(i,:);
            end
        end
        
        a = 2 - l * (2 / Max_iter); % 线性收敛因子
        
        % 更新位置
        for i = 1:SearchAgents_no
            for j = 1:dim
                % 狼 A (Alpha)
                r1 = rand(); r2 = rand();
                A1 = 2 * a * r1 - a; C1 = 2 * r2;
                D_alpha = abs(C1 * Alpha_pos(j) - Positions(i,j));
                X1 = Alpha_pos(j) - A1 * D_alpha;
                
                % 狼 B (Beta)
                r1 = rand(); r2 = rand();
                A2 = 2 * a * r1 - a; C2 = 2 * r2;
                D_beta = abs(C2 * Beta_pos(j) - Positions(i,j));
                X2 = Beta_pos(j) - A2 * D_beta;
                
                % 狼 C (Delta)
                r1 = rand(); r2 = rand();
                A3 = 2 * a * r1 - a; C3 = 2 * r2;
                D_delta = abs(C3 * Delta_pos(j) - Positions(i,j));
                X3 = Delta_pos(j) - A3 * D_delta;
                
                Positions(i,j) = (X1 + X2 + X3) / 3;
            end
        end
        Convergence_curve(l) = Alpha_score;
    end
end

function score = internal_fitness(x, S)
    w1 = 0.001; w2 = 1.0; % 距离权重与均衡度权重
    x(x < 0.01) = 0.01;
    X = ceil(x .* S.P); 
    total_dist = S.alldis;
    Y = S.YFenPei; 
    for i = 1:length(X)
        hID = S.DFenPei{i}(1); eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12; 
    end
    score = w1 * total_dist + w2 * var(Y); % 综合目标：距离最小且人数分配均衡
end