%% 三算法疏散分配对比 - 一键运行版
%  包含 GA / DE / NSGA-II 三种算法，最终输出叠加对比图
%  依赖数据文件: sj5.mat（与本脚本放同一目录）
%
%  输出图:
%    图1  GA 收敛曲线
%    图2  DE 收敛曲线
%    图3  NSGA-II 帕累托前沿
%    图4  三算法叠加对比地图（核心输出）
%
clc; clear; close all; tic;

% ======================================================================
%  0. 加载数据
% ======================================================================
if ~exist('sj5.mat','file')
    error('未找到数据文件 sj5.mat，请将其与本脚本放在同一目录。');
end
load('sj5.mat');

% 坐标偏移（统一绘图基准）
raw_x    = data.start(:,1);   raw_y  = data.start(:,2);
offset_x = min(raw_x);        offset_y = min(raw_y);
house_x  = raw_x - offset_x;  house_y  = raw_y - offset_y;
binan_x  = data.binan(:,1) - offset_x;
binan_y  = data.binan(:,2) - offset_y;

% 适配字段
if exist('dis','var'), data.dis = dis; end
data.P = cellfun(@(x) length(x)-1, data.DFenPei);

% 固定分配点预处理（三算法共用）
data.alldis  = 0;
data.YFenPei = zeros(1, size(data.binan,1));
data.FID     = [];
for k = 1:length(B)
    if length(B{k}) == 1
        tb = B{k};
        data.YFenPei(tb) = data.YFenPei(tb) + 12;
        data.FID  = [data.FID; k, tb];
        data.alldis = data.alldis + data.dis(k, tb);
    end
end
nVar = length(data.DFenPei);

% 权重（三算法统一）
weights.w1 = 0.001;
weights.w2 = 1.0;

fprintf('数据加载完毕，灵活分配点: %d，固定分配点: %d\n', nVar, size(data.FID,1));

% ======================================================================
%  1. GA 遗传算法
% ======================================================================
fprintf('\n========== [1/3] GA 遗传算法 ==========\n');

ga_popSize = 80;
ga_maxGen  = 1000;
ga_pc      = 0.85;
ga_pm      = 0.20;

GApop = rand(ga_popSize, nVar);
ga_fit = zeros(ga_popSize, 1);
for i = 1:ga_popSize
    ga_fit(i) = fitness_fn(GApop(i,:), data, weights);
end
[ga_bestScore, idx] = min(ga_fit);
ga_zbest   = GApop(idx,:);
ga_curve   = zeros(1, ga_maxGen);

for g = 1:ga_maxGen
    % 锦标赛选择
    newPop = zeros(size(GApop));
    for i = 1:ga_popSize
        cand = randi(ga_popSize,[1,3]);
        [~,w] = min(ga_fit(cand));
        newPop(i,:) = GApop(cand(w),:);
    end
    % 单点交叉
    for i = 1:2:ga_popSize
        if rand < ga_pc
            cp = randi(nVar);
            tmp = newPop(i,cp:end);
            newPop(i,cp:end)   = newPop(i+1,cp:end);
            newPop(i+1,cp:end) = tmp;
        end
    end
    % 高斯变异
    for i = 1:ga_popSize
        if rand < ga_pm
            mp = randi(nVar);
            newPop(i,mp) = newPop(i,mp) + 0.2*randn();
        end
    end
    newPop = min(max(newPop,0),1);
    GApop  = newPop;
    % 精英保留
    for i = 1:ga_popSize
        ga_fit(i) = fitness_fn(GApop(i,:), data, weights);
        if ga_fit(i) < ga_bestScore
            ga_bestScore = ga_fit(i);
            ga_zbest     = GApop(i,:);
        end
    end
    GApop(1,:) = ga_zbest;
    ga_fit(1)  = ga_bestScore;
    ga_curve(g) = ga_bestScore;
    if mod(g,50)==0
        fprintf('  GA 第 %d 代，最优适应度: %.4f\n', g, ga_bestScore);
    end
end

ga_zbest(ga_zbest<0.01) = 0.01;
X_GA    = ceil(ga_zbest .* data.P);
eID_GA  = zeros(1, nVar);
for i = 1:nVar
    eID_GA(i) = data.DFenPei{i}(X_GA(i)+1);
end
fprintf('GA 完成，最优适应度: %.4f\n', ga_bestScore);

% ======================================================================
%  2. DE 差分进化
% ======================================================================
fprintf('\n========== [2/3] DE 差分进化 ==========\n');

de_popSize = 60;
de_maxGen  = 1000;
de_F       = 0.5;   % 缩放因子
de_CR      = 0.3;   % 交叉概率

DEpop = rand(de_popSize, nVar);
de_fit = zeros(de_popSize, 1);
for i = 1:de_popSize
    de_fit(i) = fitness_fn(DEpop(i,:), data, weights);
end
[de_bestScore, idx] = min(de_fit);
de_zbest  = DEpop(idx,:);
de_curve  = zeros(1, de_maxGen);

for g = 1:de_maxGen
    for i = 1:de_popSize
        % 随机选三个不同个体
        others = randperm(de_popSize, 4);
        others(others==i) = [];
        a = others(1); b_idx = others(2); c = others(3);
        % 变异向量
        v = DEpop(a,:) + de_F * (DEpop(b_idx,:) - DEpop(c,:));
        v = min(max(v,0),1);
        % 二项交叉
        trial = DEpop(i,:);
        j0    = randi(nVar);
        mask  = (rand(1,nVar) < de_CR);
        mask(j0) = true;
        trial(mask) = v(mask);
        % 贪婪选择
        f_trial = fitness_fn(trial, data, weights);
        if f_trial <= de_fit(i)
            DEpop(i,:) = trial;
            de_fit(i)  = f_trial;
            if f_trial < de_bestScore
                de_bestScore = f_trial;
                de_zbest     = trial;
            end
        end
    end
    de_curve(g) = de_bestScore;
    if mod(g,50)==0
        fprintf('  DE 第 %d 代，最优适应度: %.4f\n', g, de_bestScore);
    end
end

de_zbest(de_zbest<0.01) = 0.01;
X_DE    = ceil(de_zbest .* data.P);
eID_DE  = zeros(1, nVar);
for i = 1:nVar
    eID_DE(i) = data.DFenPei{i}(X_DE(i)+1);
end
fprintf('DE 完成，最优适应度: %.4f\n', de_bestScore);

% ======================================================================
%  3. NSGA-II 多目标优化
% ======================================================================
fprintf('\n========== [3/3] NSGA-II 多目标优化 ==========\n');

% 检查 gamultiobj 是否可用
has_gamultiobj = ~isempty(which('gamultiobj'));

if has_gamultiobj
    % --- 使用 MATLAB 自带 gamultiobj ---
    nsga_popSize = 100;
    nsga_maxGen  = 1000;
    opts = optimoptions('gamultiobj', ...
        'PopulationSize', nsga_popSize, ...
        'MaxGenerations', nsga_maxGen, ...
        'Display',        'iter', ...
        'FunctionTolerance', 0, ...
        'PlotFcn',        []);
    lb = zeros(1,nVar); ub = ones(1,nVar);
    obj_fn = @(x) nsga_obj(x, data);
    [xSol, fSol] = gamultiobj(obj_fn, nVar, [],[],[],[], lb, ub, [], opts);

    % 选帕累托前沿中总距离最短的解
    [~, bestIdx]  = min(fSol(:,1));
    nsga_zbest    = xSol(bestIdx,:);
    nsga_fSol     = fSol;          % 保存帕累托数据用于绘图
    nsga_bestScore = fSol(bestIdx,1) * weights.w1 + fSol(bestIdx,2) * weights.w2;
else
    % --- 内置轻量级 NSGA-II（无需工具箱） ---
    fprintf('  未检测到 Optimization Toolbox，使用内置 NSGA-II...\n');
    nsga_popSize = 80;
    nsga_maxGen  = 1000;
    [nsga_zbest, nsga_fSol] = run_nsga2(nVar, nsga_popSize, nsga_maxGen, data);
    nsga_bestScore = fitness_fn(nsga_zbest, data, weights);
end

nsga_zbest(nsga_zbest<0.01) = 0.01;
X_NSGA    = ceil(nsga_zbest .* data.P);
eID_NSGA  = zeros(1, nVar);
for i = 1:nVar
    eID_NSGA(i) = data.DFenPei{i}(X_NSGA(i)+1);
end
fprintf('NSGA-II 完成\n');

% ======================================================================
%  4. 差异统计
% ======================================================================
diff_DE   = (eID_DE   ~= eID_GA);
diff_NSGA = (eID_NSGA ~= eID_GA);

fprintf('\n========== 差异统计 ==========\n');
fprintf('灵活分配点总数    : %d\n', nVar);
fprintf('DE   与 GA 不同   : %d 条 (%.1f%%)\n', sum(diff_DE),   100*mean(diff_DE));
fprintf('NSGA 与 GA 不同   : %d 条 (%.1f%%)\n', sum(diff_NSGA), 100*mean(diff_NSGA));

% ======================================================================
%  5. 图1 - GA 收敛曲线
% ======================================================================
figure('Color','w','Name','GA Convergence','Position',[50,550,560,380]);
plot(ga_curve,'LineWidth',2,'Color',[0.85,0.33,0.10]);
grid on; box on;
xlabel('Iteration','FontWeight','bold');
ylabel('Best Fitness','FontWeight','bold');
title('GA Convergence Curve','FontSize',12,'FontWeight','bold');
xlim([1, ga_maxGen]);

% ======================================================================
%  图2 - DE 收敛曲线
% ======================================================================
figure('Color','w','Name','DE Convergence','Position',[630,550,560,380]);
plot(de_curve,'LineWidth',2,'Color',[0.55,0.10,0.90]);
grid on; box on;
xlabel('Iteration','FontWeight','bold');
ylabel('Best Fitness','FontWeight','bold');
title('DE Convergence Curve','FontSize',12,'FontWeight','bold');
xlim([1, de_maxGen]);

% ======================================================================
%  图3 - NSGA-II 帕累托前沿（折线版，与收敛曲线风格统一）
% ======================================================================
figure('Color','w','Name','NSGA-II Pareto Front','Position',[50,100,560,380]);
% 按F1排序后连线，使折线连贯
[sorted_f, sort_idx] = sortrows(nsga_fSol, 1);
plot(sorted_f(:,1), sorted_f(:,2), ...
     'LineWidth',2,'Color',[0.42,0.78,0.60]);
grid on; box on;
xlabel('F1: Total Evacuation Distance','FontWeight','bold');
ylabel('F2: Capacity Variance (Balance)','FontWeight','bold');
title('NSGA-II Pareto Optimal Front','FontSize',12,'FontWeight','bold');

% ======================================================================
%  图4 - 三算法叠加对比地图（核心输出）
% ======================================================================
figure('Color','w','Name','Algorithm Comparison Map','Position',[630,50,1000,900]);
hold on; box on;

% 道路底图
if isfield(data,'road')
    for i = 1:length(data.road)
        plot(data.road{i}(:,1)-offset_x, data.road{i}(:,2)-offset_y, ...
             'Color',[0.88,0.88,0.88],'LineWidth',0.5);
    end
end

% --- GA 固定分配线（红色） ---
if ~isempty(data.FID)
    for i = 1:size(data.FID,1)
        line([house_x(data.FID(i,1)), binan_x(data.FID(i,2))], ...
             [house_y(data.FID(i,1)), binan_y(data.FID(i,2))], ...
             'Color',[0.90, 0.10, 0.10, 0.45],'LineWidth',0.8);
    end
end

% --- GA 灵活分配线（红色，所有点，先铺底） ---
for i = 1:nVar
    hIdx = data.DFenPei{i}(1);
    line([house_x(hIdx), binan_x(eID_GA(i))], ...
         [house_y(hIdx), binan_y(eID_GA(i))], ...
         'Color',[0.90, 0.10, 0.10, 0.45],'LineWidth',0.8);
end

% --- DE 与 GA 不同的分配线（紫色） ---
h_de_diff = [];
for i = find(diff_DE)
    hIdx = data.DFenPei{i}(1);
    h = line([house_x(hIdx), binan_x(eID_DE(i))], ...
             [house_y(hIdx), binan_y(eID_DE(i))], ...
             'Color',[0.55, 0.10, 0.90, 0.35],'LineWidth',1.0);
    if isempty(h_de_diff), h_de_diff = h; end
end

% --- NSGA-II 与 GA 不同的分配线（绿色） ---
h_nsga_diff = [];
for i = find(diff_NSGA)
    hIdx = data.DFenPei{i}(1);
    h = line([house_x(hIdx), binan_x(eID_NSGA(i))], ...
             [house_y(hIdx), binan_y(eID_NSGA(i))], ...
             'Color',[0.42, 0.78, 0.60, 0.65],'LineWidth',1.0);
    if isempty(h_nsga_diff), h_nsga_diff = h; end
end

% --- 节点（最上层） ---
h_res = scatter(house_x, house_y, 9,  [0.00,0.18,0.60], 'filled');
h_shl = scatter(binan_x, binan_y, 80, 'g', '^', 'filled', ...
                'MarkerEdgeColor','k','LineWidth',0.8);

% --- 格式化 ---
axis equal; axis tight; grid on;
ax = gca;
ax.LooseInset  = ax.TightInset;
ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
ax.FontSize    = 11;  ax.FontWeight = 'bold';
margin = 100;
xlim([min(house_x)-margin, max(house_x)+margin]);
ylim([min(house_y)-margin, max(house_y)+margin]);

title('GA / DE / NSGA-II Allocation Comparison Map', ...
      'FontSize',14,'FontWeight','bold');
xlabel('X Coordinate Offset (m)','FontWeight','bold');
ylabel('Y Coordinate Offset (m)','FontWeight','bold');

% --- 图例 ---
h_ga_leg   = line(NaN,NaN,'Color',[0.90,0.10,0.10,0.80],'LineWidth',2.0);
h_de_leg   = line(NaN,NaN,'Color',[0.55,0.10,0.90,0.50],'LineWidth',2.0);
h_nsga_leg = line(NaN,NaN,'Color',[0.42,0.78,0.60,0.80],'LineWidth',2.0);

legend([h_shl, h_res, h_ga_leg, h_de_leg, h_nsga_leg], ...
    {'Shelter', ...
     'Residential', ...
     'GA allocation', ...
     'DE allocation (differs from GA)', ...
     'NSGA-II allocation (differs from GA)'}, ...
    'Location','northeast','FontSize',7,'Box','on');

fprintf('\n全部完成！共耗时 %.1f 秒。\n', toc);
fprintf('\n图例说明:\n');
fprintf('  红色线   = GA 全部分配（底图基准）\n');
fprintf('  紫色粗线 = DE 与 GA 结果不同的分配连线\n');
fprintf('  绿色粗线 = NSGA-II 与 GA 结果不同的分配连线\n');
fprintf('  绿色三角 = 避难场所\n');
fprintf('  蓝色圆点 = 住宅点位\n');


% ======================================================================
%  本地函数区
% ======================================================================

% ---------- 共用适应度函数 ----------
function score = fitness_fn(x, S, w)
    x(x<0.01) = 0.01;
    X = ceil(x .* S.P);
    total_dist = S.alldis;
    Y = S.YFenPei;
    for i = 1:length(X)
        hID = S.DFenPei{i}(1);
        eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12;
    end
    score = w.w1 * total_dist + w.w2 * var(Y);
end

% ---------- NSGA-II 双目标函数（供 gamultiobj 调用） ----------
function f = nsga_obj(x, S)
    x(x<0.01) = 0.01;
    X = ceil(x .* S.P);
    total_dist = S.alldis;
    Y = S.YFenPei;
    for i = 1:length(X)
        hID = S.DFenPei{i}(1);
        eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12;
    end
    f = [total_dist, var(Y)];
end

% ---------- 内置轻量级 NSGA-II（无需工具箱） ----------
function [best_x, pareto_f] = run_nsga2(nVar, popSize, maxGen, S)
    pop = rand(popSize, nVar);
    for g = 1:maxGen
        % 生成子代（SBX交叉 + 多项式变异，简化版）
        child = pop;
        for i = 1:2:popSize
            j = mod(i,popSize)+1;
            if rand < 0.9
                alpha = rand(1,nVar);
                child(i,:)   = alpha.*pop(i,:)   + (1-alpha).*pop(j,:);
                child(j,:)   = alpha.*pop(j,:)   + (1-alpha).*pop(i,:);
            end
        end
        % 变异
        for i = 1:popSize
            if rand < 0.15
                mp = randi(nVar);
                child(i,mp) = child(i,mp) + 0.1*randn();
            end
        end
        child = min(max(child,0),1);
        % 合并父子代
        combined = [pop; child];
        N_comb   = size(combined,1);
        % 计算目标值
        F_obj = zeros(N_comb,2);
        for i = 1:N_comb
            F_obj(i,:) = nsga_obj(combined(i,:), S);
        end
        % 非支配排序（快速版）
        rank_vec = fast_nondom_sort(F_obj);
        % 拥挤距离（对第一前沿）
        cd_vec = crowding_dist(F_obj, rank_vec, 1);
        % 选择 popSize 个个体
        [~, sel_idx] = sortrows([rank_vec(:), -cd_vec(:)]);
        pop = combined(sel_idx(1:popSize),:);
    end
    % 最终帕累托前沿
    F_final   = zeros(popSize,2);
    for i = 1:popSize
        F_final(i,:) = nsga_obj(pop(i,:), S);
    end
    rank_final = fast_nondom_sort(F_final);
    front1_idx = find(rank_final==1);
    pareto_f   = F_final(front1_idx,:);
    % 选总距离最短的解
    [~, bi]  = min(pareto_f(:,1));
    best_x   = pop(front1_idx(bi),:);
end

% ---------- 快速非支配排序 ----------
function rank_vec = fast_nondom_sort(F)
    N = size(F,1);
    rank_vec = zeros(N,1);
    dom_count = zeros(N,1);
    dom_set   = cell(N,1);
    for i = 1:N
        for j = 1:N
            if i==j, continue; end
            if all(F(i,:) <= F(j,:)) && any(F(i,:) < F(j,:))
                dom_set{i}(end+1) = j;
            elseif all(F(j,:) <= F(i,:)) && any(F(j,:) < F(i,:))
                dom_count(i) = dom_count(i)+1;
            end
        end
        if dom_count(i)==0, rank_vec(i)=1; end
    end
    cur_rank = 1;
    while any(rank_vec==cur_rank)
        next_front = [];
        for i = find(rank_vec==cur_rank)'
            for j = dom_set{i}
                dom_count(j) = dom_count(j)-1;
                if dom_count(j)==0
                    rank_vec(j) = cur_rank+1;
                    next_front(end+1) = j; %#ok
                end
            end
        end
        cur_rank = cur_rank+1;
        if cur_rank > N, break; end
    end
end

% ---------- 拥挤距离计算 ----------
function cd = crowding_dist(F, rank_vec, target_rank)
    idx = find(rank_vec==target_rank);
    n   = length(idx);
    cd  = zeros(size(F,1),1);
    if n <= 2
        cd(idx) = Inf;
        return;
    end
    f_sub = F(idx,:);
    nObj  = size(F,2);
    d     = zeros(n,1);
    for m = 1:nObj
        [~, order] = sort(f_sub(:,m));
        fmin = f_sub(order(1),m);
        fmax = f_sub(order(end),m);
        if fmax == fmin, continue; end
        d(order(1))   = Inf;
        d(order(end)) = Inf;
        for k = 2:n-1
            d(order(k)) = d(order(k)) + ...
                (f_sub(order(k+1),m) - f_sub(order(k-1),m)) / (fmax-fmin);
        end
    end
    cd(idx) = d;
end
