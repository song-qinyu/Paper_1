%% FA 萤火虫算法疏散分配优化 - 自动评价指标版
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

% 确保核心字段存在
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

% ========================== 2. FA 参数设置 ==========================
nVar = length(data.DFenPei);    
popSize = 30;                   
maxGen = 100;                   
alpha = 0.5;                    % 随机步长参数
beta0 = 1.0;                    % 最大吸引度
gamma = 1.0;                    % 光强吸收系数
weights.w1 = 0.001; weights.w2 = 1.0; 

% 初始化萤火虫
pop = rand(popSize, nVar);
Lightn = zeros(popSize, 1);
for i = 1:popSize
    Lightn(i) = fa_obj(pop(i,:), data, weights);
end

cg_curve = zeros(1, maxGen);

% ========================== 3. 执行 FA 进化 ==========================
fprintf('FA 优化启动，正在计算 8 项评价指标...\n');

for g = 1:maxGen
    for i = 1:popSize
        for j = 1:popSize
            if Lightn(j) < Lightn(i) % 如果 j 比 i 更亮（更优），则 i 向 j 移动
                r = norm(pop(i,:) - pop(j,:));
                beta = beta0 * exp(-gamma * r^2);
                % 移动逻辑
                pop(i,:) = pop(i,:) + beta * (pop(j,:) - pop(i,:)) + alpha * (rand(1, nVar) - 0.5);
                % 边界限制
                pop(i, pop(i,:)>1) = 1; pop(i, pop(i,:)<0.01) = 0.01;
                % 更新亮度
                Lightn(i) = fa_obj(pop(i,:), data, weights);
            end
        end
    end
    
    [fMin, bestIdx] = min(Lightn);
    bestPos = pop(bestIdx, :);
    cg_curve(g) = fMin;
end

% ========================== 4. 计算 8 项评价指标 ==========================
MET = toc; 
zbest = bestPos;
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P); 

% 路径数据提取
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

% 指标统计
TED = sum(all_dist);           
ATD = mean(all_dist);          
MID = max(all_dist);           
BTV = fMin;                    
SUR = (length(unique([dynamic_used_bins, data.FID(:,2)'])) / size(data.binan, 1)) * 100; 

% 指标 6: 收敛代数 (CG)
change = abs(diff(cg_curve));
last_c = find(change > 1e-6, 1, 'last');
if isempty(last_c), CG = 1; else CG = last_c + 1; end
SD = 0; % 稳定性 (单次运行设为0)

% ========================== 5. 输出结果面板 ==========================
fprintf('\n==============================================\n');
fprintf('   FA (Firefly Algorithm) 性能评价指标\n');
fprintf('==============================================\n');
fprintf('TED (总距离):   %.2f m\n', TED);
fprintf('ATD (平均距离): %.2f m\n', ATD);
fprintf('MID (最大距离): %.2f m\n', MID);
fprintf('SUR (利用 rate): %.2f %%\n', SUR);
fprintf('BTV (最佳适应度): %.6f\n', BTV);
fprintf('MET (执行时间): %.4f s\n', MET);
fprintf('CG  (收敛代数): %d\n', CG);
fprintf('SD  (稳定性):   %.4f\n', SD);
fprintf('==============================================\n');

% 绘制收敛曲线
figure('Color','w'); plot(cg_curve, 'Color', [0.85, 0.33, 0.1], 'LineWidth', 2);
grid on; xlabel('Iteration'); ylabel('Best Fitness');
title('FA Optimization Convergence');

% ========================== 适应度函数 ==========================
function score = fa_obj(x, S, w)
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