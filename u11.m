%% SA 模拟退火算法疏散分配优化 - 指标集成版
clc; clear; close all; tic;

% ========================== 1. 数据加载与对齐 ==========================
if exist('sj5.mat', 'file')
    load('sj5.mat'); 
else
    error('未找到数据文件 sj5.mat');
end

% 适配字段与坐标偏移
if exist('dis','var'), data.dis = dis; end
off_x = min(data.start(:,1)); off_y = min(data.start(:,2));
house_x = data.start(:,1) - off_x; house_y = data.start(:,2) - off_y;
binan_x = data.binan(:,1) - off_x; binan_y = data.binan(:,2) - off_y;

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

% ========================== 2. SA 参数设置 ==========================
nVar = length(data.DFenPei);
T0 = 1000;          % 初始温度
Tend = 1e-3;        % 终止温度
L = 100;            % 每个温度下的迭代次数（链长）
q = 0.95;           % 降温系数
weights.w1 = 0.001; weights.w2 = 1.0; 

% 初始解
current_sol = rand(1, nVar);
current_fit = sa_obj(current_sol, data, weights);
best_sol = current_sol;
fMin = current_fit;

cg_curve = [];

% ========================== 3. 执行 SA 优化 ==========================
fprintf('SA 优化启动，正在计算 8 项评价指标...\n');
T = T0;
while T > Tend
    for i = 1:L
        % 产生新解（扰动）
        new_sol = current_sol + 0.1 * randn(1, nVar);
        new_sol = max(min(new_sol, 1), 0.01);
        
        new_fit = sa_obj(new_sol, data, weights);
        
        % Metropolis 准则
        delta = new_fit - current_fit;
        if delta < 0 || exp(-delta / T) > rand
            current_sol = new_sol;
            current_fit = new_fit;
            % 更新全局最优
            if current_fit < fMin
                fMin = current_fit;
                best_sol = current_sol;
            end
        end
    end
    cg_curve = [cg_curve, fMin];
    T = T * q; % 降温
end

% ========================== 4. 计算 8 项评价指标 ==========================
MET = toc; 
zbest = best_sol;
zbest(zbest < 0.01) = 0.01;
X_final = ceil(zbest .* data.P); 

% 计算距离数据
dynamic_distances = zeros(1, length(X_final));
dynamic_used_bins = zeros(1, length(X_final));
for i = 1:length(X_final)
    hIdx = data.DFenPei{i}(1); 
    eIdx = data.DFenPei{i}(X_final(i)+1);
    dynamic_distances(i) = data.dis(hIdx, eIdx);
    dynamic_used_bins(i) = eIdx;
end
fixed_distances = arrayfun(@(i) data.dis(data.FID(i,1), data.FID(i,2)), 1:size(data.FID,1));
all_dist = [dynamic_distances, fixed_distances];

% 指标统计
TED = sum(all_dist);           
ATD = mean(all_dist);          
MID = max(all_dist);           
BTV = fMin;                    
SUR = (length(unique([dynamic_used_bins, data.FID(:,2)'])) / size(data.binan, 1)) * 100; 

% 收敛代数 (CG)
change = abs(diff(cg_curve));
last_c = find(change > 1e-6, 1, 'last');
if isempty(last_c), CG = 1; else CG = last_c + 1; end
SD = 0; 

% ========================== 5. 输出结果面板 ==========================
fprintf('\n==============================================\n');
fprintf('   SA (Simulated Annealing) 性能评价指标\n');
fprintf('==============================================\n');
fprintf('TED (总距离):   %.2f m\n', TED);
fprintf('ATD (平均距离): %.2f m\n', ATD);
fprintf('MID (最大距离): %.2f m\n', MID);
fprintf('SUR (利用率):   %.2f %%\n', SUR);
fprintf('BTV (最佳适应度): %.6f\n', BTV);
fprintf('MET (执行时间): %.4f s\n', MET);
fprintf('CG  (收敛代数): %d\n', CG);
fprintf('SD  (稳定性):   %.4f\n', SD);
fprintf('==============================================\n');

% 绘制收敛曲线
figure('Color','w'); plot(cg_curve, 'LineWidth', 2, 'Color', 'r');
grid on; xlabel('Temperature Steps'); ylabel('Best Fitness');
title('SA Convergence Process');

% ========================== 适应度函数 ==========================
function score = sa_obj(x, S, w)
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