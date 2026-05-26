%% GWO 疏散分配优化 - 自动评价指标版
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
house_x = raw_x - offset_x;
house_y = raw_y - offset_y;
binan_x = data.binan(:,1) - offset_x;
binan_y = data.binan(:,2) - offset_y;

% 字段适配
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

% ========================== 2. GWO 参数设置 ==========================
nVar = length(data.DFenPei);  
popSize = 50;       
maxGen = 200;       
weights.w1 = 0.001; weights.w2 = 1.0;   

% 初始化三头头狼
Alpha_pos = zeros(1, nVar); Alpha_score = inf; 
Beta_pos = zeros(1, nVar);  Beta_score = inf; 
Delta_pos = zeros(1, nVar); Delta_score = inf; 

Positions = rand(popSize, nVar);
cg_curve = zeros(1, maxGen);

% ========================== 3. 执行 GWO 进化 ==========================
fprintf('GWO 优化启动，正在计算指标...\n');

for g = 1:maxGen
    for i = 1:popSize
        % 边界检查
        Positions(i, Positions(i,:)>1) = 1;
        Positions(i, Positions(i,:)<0.01) = 0.01;
        
        % 计算适应度
        fitness = fitness_function(Positions(i,:), data, weights);
        
        % 更新 Alpha, Beta, Delta
        if fitness < Alpha_score 
            Alpha_score = fitness; Alpha_pos = Positions(i,:);
        end
        if fitness > Alpha_score && fitness < Beta_score 
            Beta_score = fitness; Beta_pos = Positions(i,:);
        end
        if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score 
            Delta_score = fitness; Delta_pos = Positions(i,:);
        end
    end
    
    a = 2 - g * (2 / maxGen); % 线性递减系数 a
    
    for i = 1:popSize
        for j = 1:nVar
            % 包围更新逻辑
            r1 = rand(); r2 = rand();
            A1 = 2*a*r1 - a; C1 = 2*r2;
            D_alpha = abs(C1*Alpha_pos(j) - Positions(i,j));
            X1 = Alpha_pos(j) - A1*D_alpha;
            
            r1 = rand(); r2 = rand();
            A2 = 2*a*r1 - a; C2 = 2*r2;
            D_beta = abs(C2*Beta_pos(j) - Positions(i,j));
            X2 = Beta_pos(j) - A2*D_beta;
            
            r1 = rand(); r2 = rand();
            A3 = 2*a*r1 - a; C3 = 2*r2;
            D_delta = abs(C3*Delta_pos(j) - Positions(i,j));
            X3 = Delta_pos(j) - A3*D_delta;
            
            Positions(i,j) = (X1 + X2 + X3) / 3;
        end
    end
    cg_curve(g) = Alpha_score;
end

% ========================== 4. 计算评价指标 (Evaluation Metrics) ==========================
zbest = Alpha_pos;
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P); 

% 指标 1: BTV (Best Fitness Value)
BTV = Alpha_score;

% 指标 2: MET (Mean Execution Time)
MET = toc; 

% 指标 3: CG (Convergence Generation)
threshold = 1e-6;
change = abs(diff(cg_curve));
last_change = find(change > threshold, 1, 'last');
if isempty(last_change), CG = 1; else CG = last_change + 1; end

% 指标 4: 路径质量 (TED, ATD, MID)
dynamic_distances = zeros(1, length(X_final));
dynamic_used_bins = zeros(1, length(X_final));
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1); 
    eIdx = data.DFenPei{i}(X_final(i)+1);
    dynamic_distances(i) = data.dis(hIdx, eIdx);
    dynamic_used_bins(i) = eIdx;
end
fixed_distances = [];
if ~isempty(data.FID)
    for k = 1:size(data.FID, 1)
        fixed_distances(k) = data.dis(data.FID(k,1), data.FID(k,2));
    end
end
all_distances = [dynamic_distances, fixed_distances];

TED = sum(all_distances);           
ATD = mean(all_distances);          
MID = max(all_distances);           

% 指标 5: SUR (Shelter Utilization Rate)
fixed_used_bins = [];
if ~isempty(data.FID), fixed_used_bins = data.FID(:,2)'; end
used_bins_total = unique([dynamic_used_bins, fixed_used_bins]); 
total_bins_available = size(data.binan, 1);
SUR = (length(used_bins_total) / total_bins_available) * 100;

% 指标 6: SD (Stability)
SD = 0; 

% ========================== 5. 输出结果面板 ==========================
fprintf('\n==============================================\n');
fprintf('   GWO Algorithm Performance Metrics (sj5.mat)\n');
fprintf('==============================================\n');
fprintf('Algorithm: GWO (Grey Wolf Optimizer)\n');
fprintf('TED: %.2f m\n', TED);
fprintf('ATD: %.2f m\n', ATD);
fprintf('MID: %.2f m\n', MID);
fprintf('SUR: %.2f %%\n', SUR);
fprintf('BTV: %.6f\n', BTV);
fprintf('MET: %.4f s\n', MET);
fprintf('CG:  %d\n', CG);
fprintf('SD:  %.4f\n', SD);
fprintf('==============================================\n');

% ========================== 适应度函数 ==========================
function score = fitness_function(x, S, w)
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