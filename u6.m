%% SSA 疏散分配优化 - 统一规格 + 8项指标集成版
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

% 适配字段
if exist('dis','var'), data.dis = dis; end
data.P = cellfun(@(x) length(x)-1, data.DFenPei);

% 初始化固定分配
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

% ========================== 2. SSA 参数设置 ==========================
pop = 50;           % 种群数量
M = 150;            % 最大迭代次数
dim = length(data.DFenPei); 
lb = 0; ub = 1;
P_percent = 0.2;    % 发现者比例
weights.w1 = 0.001; weights.w2 = 1.0; 
x = rand(pop, dim); 
fit = zeros(1, pop);
for i = 1:pop
    fit(i) = ssa_obj(x(i,:), data, weights);
end
pFit = fit; pX = x; 
[fMin, bestI] = min(fit);
bestX = x(bestI, :);
Convergence_curve = zeros(1, M);

% ========================== 3. 执行 SSA 优化 ==========================
fprintf('SSA 优化启动，正在计算 8 项评价指标...\n');
for t = 1:M
    [~, sortIndex] = sort(fit);
    [fmax, B_idx] = max(fit);
    worse = x(B_idx, :);
    
    r2 = rand;
    % 更新发现者
    for i = 1:round(pop * P_percent)
        if r2 < 0.8
            x(sortIndex(i),:) = pX(sortIndex(i),:) * exp(-i/(rand*M));
        else
            x(sortIndex(i),:) = pX(sortIndex(i),:) + randn(1,dim);
        end
    end
    
    % 更新加入者
    for i = (round(pop * P_percent) + 1):pop
        if i > pop/2
            x(sortIndex(i),:) = randn * exp((worse - pX(sortIndex(i),:)) / (i^2));
        else
            A = floor(rand(1, dim) * 2) * 2 - 1;
            L = A .* ( (A * A')^-1 );
            x(sortIndex(i),:) = bestX + abs(pX(sortIndex(i),:) - bestX) .* L;
        end
    end
    
    % 边界检查与适应度更新
    for i = 1:pop
        x(i,:) = max(min(x(i,:), ub), lb);
        fit(i) = ssa_obj(x(i,:), data, weights);
        if fit(i) < pFit(i)
            pFit(i) = fit(i); pX(i,:) = x(i,:);
        end
        if pFit(i) < fMin
            fMin = pFit(i); bestX = pX(i,:);
        end
    end
    Convergence_curve(t) = fMin;
end

% ========================== 4. 计算 8 项评价指标 ==========================
MET = toc; % MET: Mean Execution Time
zbest = bestX;
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P); 

% 计算各项距离 (指标核心数据)
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
all_dist = [dynamic_distances, fixed_distances];

% 8项指标赋值
TED = sum(all_dist);            % 指标1: 总疏散距离
ATD = mean(all_dist);           % 指标2: 平均疏散距离
MID = max(all_dist);            % 指标3: 最大疏散距离
BTV = fMin;                     % 指标4: 最佳适应度值
SUR = (length(unique([dynamic_used_bins, data.FID(:,2)'])) / size(data.binan, 1)) * 100; % 指标5: 避难所利用率

% 指标6: 收敛代数 (CG)
change = abs(diff(Convergence_curve));
last_c = find(change > 1e-6, 1, 'last');
if isempty(last_c), CG = 1; else CG = last_c + 1; end
SD = 0; % 指标7: 稳定性 (单次运行设为0)

% 输出指标面板
fprintf('\n==============================================\n');
fprintf('   SSA 算法性能评价指标 (基于 sj5.mat)\n');
fprintf('==============================================\n');
fprintf('TED (总距离):   %.2f m\n', TED);
fprintf('ATD (平均距离): %.2f m\n', ATD);
fprintf('MID (最大距离): %.2f m\n', MID);
fprintf('SUR (利用率):   %.2f %%\n', SUR);
fprintf('BTV (适应度):   %.6f\n', BTV);
fprintf('MET (执行时间): %.4f s\n', MET);
fprintf('CG  (收敛代数): %d\n', CG);
fprintf('SD  (稳定性):   %.4f\n', SD);
fprintf('==============================================\n');

% ========================== 5. 结果可视化 ==========================
% (此处保留你原有的绘图代码，包括 2D 地图、收敛曲线和 3D 地形图)
% ... [为了长度简略，运行时请接上你原有的绘图代码部分] ...

% ========================== 适应度函数 ==========================
function score = ssa_obj(x, S, w)
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