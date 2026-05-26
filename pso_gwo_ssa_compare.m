%% 三算法疏散分配对比 - PSO / GWO / SSA 一键运行版
%  以 PSO 为底图基准，叠加 GWO（紫色）和 SSA（春日青）的差异分配线
%
%  输出图:
%    图1  PSO 收敛曲线
%    图2  GWO 收敛曲线
%    图3  SSA 收敛曲线
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

raw_x    = data.start(:,1);   raw_y  = data.start(:,2);
offset_x = min(raw_x);        offset_y = min(raw_y);
house_x  = raw_x - offset_x;  house_y  = raw_y - offset_y;
binan_x  = data.binan(:,1) - offset_x;
binan_y  = data.binan(:,2) - offset_y;

if exist('dis','var'), data.dis = dis; end
data.P = cellfun(@(x) length(x)-1, data.DFenPei);

% 固定分配点预处理（三算法共用）
data.alldis_fixed  = 0;
data.YFenPei_fixed = zeros(1, size(data.binan,1));
data.FID           = [];
for k = 1:length(B)
    if length(B{k}) == 1
        tb = B{k};
        data.YFenPei_fixed(tb) = data.YFenPei_fixed(tb) + 12;
        data.FID    = [data.FID; k, tb];
        data.alldis_fixed = data.alldis_fixed + data.dis(k, tb);
    end
end
nVar = length(data.DFenPei);

weights.w1 = 0.001;
weights.w2 = 1.0;

fprintf('数据加载完毕，灵活分配点: %d，固定分配点: %d\n', nVar, size(data.FID,1));

% ======================================================================
%  1. PSO 粒子群算法（基准）
% ======================================================================
fprintf('\n========== [1/3] PSO 粒子群算法 ==========\n');

pso_nPop  = 50;
pso_MaxIt = 1000;
w_inertia = 0.8; c1 = 1.5; c2 = 1.5;

particle(1:pso_nPop) = struct('x',[],'v',[],'best_x',[],'best_cost',inf,'cost',inf);
gBest.cost = inf; gBest.x = [];

for i = 1:pso_nPop
    particle(i).x         = rand(1, nVar);
    particle(i).v         = zeros(1, nVar);
    particle(i).cost      = shared_obj(particle(i).x, data, weights);
    particle(i).best_x    = particle(i).x;
    particle(i).best_cost = particle(i).cost;
    if particle(i).cost < gBest.cost
        gBest.x    = particle(i).x;
        gBest.cost = particle(i).cost;
    end
end

pso_curve = zeros(1, pso_MaxIt);
for it = 1:pso_MaxIt
    for i = 1:pso_nPop
        particle(i).v = w_inertia*particle(i).v ...
                      + c1*rand()*(particle(i).best_x - particle(i).x) ...
                      + c2*rand()*(gBest.x - particle(i).x);
        particle(i).x    = min(max(particle(i).x + particle(i).v, 0), 1);
        particle(i).cost = shared_obj(particle(i).x, data, weights);
        if particle(i).cost < particle(i).best_cost
            particle(i).best_cost = particle(i).cost;
            particle(i).best_x    = particle(i).x;
            if particle(i).best_cost < gBest.cost
                gBest.cost = particle(i).best_cost;
                gBest.x    = particle(i).best_x;
            end
        end
    end
    pso_curve(it) = gBest.cost;
    if mod(it,100)==0
        fprintf('  PSO 第 %d 代，最优适应度: %.4f\n', it, gBest.cost);
    end
end

pso_zbest = gBest.x; pso_zbest(pso_zbest<0.01) = 0.01;
X_PSO   = ceil(pso_zbest .* data.P);
eID_PSO = zeros(1,nVar);
for i = 1:nVar, eID_PSO(i) = data.DFenPei{i}(X_PSO(i)+1); end
fprintf('PSO 完成，最优适应度: %.4f\n', gBest.cost);

% ======================================================================
%  2. GWO 灰狼优化
% ======================================================================
fprintf('\n========== [2/3] GWO 灰狼优化 ==========\n');

gwo_nPop  = 50;
gwo_MaxIt = 1000;

Alpha_pos = zeros(1,nVar); Alpha_score = inf;
Beta_pos  = zeros(1,nVar); Beta_score  = inf;
Delta_pos = zeros(1,nVar); Delta_score = inf;
Pos_gwo   = rand(gwo_nPop, nVar);
gwo_curve = zeros(1, gwo_MaxIt);

for l = 1:gwo_MaxIt
    for i = 1:gwo_nPop
        Pos_gwo(i,:) = min(max(Pos_gwo(i,:),0),1);
        fit = shared_obj(Pos_gwo(i,:), data, weights);
        if fit < Alpha_score
            Alpha_score = fit; Alpha_pos = Pos_gwo(i,:);
        elseif fit < Beta_score
            Beta_score  = fit; Beta_pos  = Pos_gwo(i,:);
        elseif fit < Delta_score
            Delta_score = fit; Delta_pos = Pos_gwo(i,:);
        end
    end
    a = 2 - l*(2/gwo_MaxIt);
    for i = 1:gwo_nPop
        for j = 1:nVar
            X1 = Alpha_pos(j) - (2*a*rand()-a)*abs(2*rand()*Alpha_pos(j)-Pos_gwo(i,j));
            X2 = Beta_pos(j)  - (2*a*rand()-a)*abs(2*rand()*Beta_pos(j) -Pos_gwo(i,j));
            X3 = Delta_pos(j) - (2*a*rand()-a)*abs(2*rand()*Delta_pos(j)-Pos_gwo(i,j));
            Pos_gwo(i,j) = (X1+X2+X3)/3;
        end
    end
    gwo_curve(l) = Alpha_score;
    if mod(l,100)==0
        fprintf('  GWO 第 %d 代，最优适应度: %.4f\n', l, Alpha_score);
    end
end

gwo_zbest = Alpha_pos; gwo_zbest(gwo_zbest<0.01) = 0.01;
X_GWO   = ceil(gwo_zbest .* data.P);
eID_GWO = zeros(1,nVar);
for i = 1:nVar, eID_GWO(i) = data.DFenPei{i}(X_GWO(i)+1); end
fprintf('GWO 完成，最优适应度: %.4f\n', Alpha_score);

% ======================================================================
%  3. SSA 麻雀搜索算法
% ======================================================================
fprintf('\n========== [3/3] SSA 麻雀搜索算法 ==========\n');

ssa_nPop  = 50;
ssa_MaxIt = 1000;
P_percent = 0.2;

x_ssa = rand(ssa_nPop, nVar);
fit_ssa = zeros(1, ssa_nPop);
for i = 1:ssa_nPop
    fit_ssa(i) = shared_obj(x_ssa(i,:), data, weights);
end
pFit = fit_ssa; pX = x_ssa;
[fMin, bestI] = min(fit_ssa);
bestX_ssa = x_ssa(bestI,:);
ssa_curve = zeros(1, ssa_MaxIt);

for t = 1:ssa_MaxIt
    [~, sortIdx] = sort(fit_ssa);
    [~, B_idx]   = max(fit_ssa);
    worse = x_ssa(B_idx,:);
    r2    = rand();
    for i = 1:round(ssa_nPop*P_percent)
        if r2 < 0.8
            x_ssa(sortIdx(i),:) = pX(sortIdx(i),:) .* exp(-i/(rand()*ssa_MaxIt));
        else
            x_ssa(sortIdx(i),:) = pX(sortIdx(i),:) + randn(1,nVar);
        end
    end
    for i = round(ssa_nPop*P_percent)+1 : ssa_nPop
        if i > ssa_nPop/2
            x_ssa(sortIdx(i),:) = randn() * exp((worse - pX(sortIdx(i),:)) / i^2);
        else
            A = floor(rand(1,nVar)*2)*2-1;
            L = A * ((A*A')^-1);
            x_ssa(sortIdx(i),:) = bestX_ssa + abs(pX(sortIdx(i),:)-bestX_ssa) .* L;
        end
    end
    for i = 1:ssa_nPop
        x_ssa(i,:) = min(max(x_ssa(i,:),0),1);
        fit_ssa(i) = shared_obj(x_ssa(i,:), data, weights);
        if fit_ssa(i) < pFit(i)
            pFit(i) = fit_ssa(i); pX(i,:) = x_ssa(i,:);
        end
        if pFit(i) < fMin
            fMin = pFit(i); bestX_ssa = pX(i,:);
        end
    end
    ssa_curve(t) = fMin;
    if mod(t,100)==0
        fprintf('  SSA 第 %d 代，最优适应度: %.4f\n', t, fMin);
    end
end

ssa_zbest = bestX_ssa; ssa_zbest(ssa_zbest<0.01) = 0.01;
X_SSA   = ceil(ssa_zbest .* data.P);
eID_SSA = zeros(1,nVar);
for i = 1:nVar, eID_SSA(i) = data.DFenPei{i}(X_SSA(i)+1); end
fprintf('SSA 完成，最优适应度: %.4f\n', fMin);

% ======================================================================
%  4. 差异统计
% ======================================================================
diff_GWO = (eID_GWO ~= eID_PSO);
diff_SSA = (eID_SSA ~= eID_PSO);

fprintf('\n========== 差异统计 ==========\n');
fprintf('灵活分配点总数       : %d\n', nVar);
fprintf('GWO 与 PSO 不同      : %d 条 (%.1f%%)\n', sum(diff_GWO), 100*mean(diff_GWO));
fprintf('SSA 与 PSO 不同      : %d 条 (%.1f%%)\n', sum(diff_SSA), 100*mean(diff_SSA));

% ======================================================================
%  图1 - PSO 收敛曲线
% ======================================================================
figure('Color','w','Name','PSO Convergence','Position',[50,600,560,360]);
plot(pso_curve,'LineWidth',2,'Color',[0.85,0.33,0.10]);
grid on; box on;
xlabel('Iteration','FontWeight','bold');
ylabel('Best Fitness','FontWeight','bold');
title('PSO Convergence Curve','FontSize',12,'FontWeight','bold');
xlim([1, pso_MaxIt]);

% ======================================================================
%  图2 - GWO 收敛曲线（紫色）
% ======================================================================
figure('Color','w','Name','GWO Convergence','Position',[630,600,560,360]);
plot(gwo_curve,'LineWidth',2,'Color',[0.55,0.10,0.90]);
grid on; box on;
xlabel('Iteration','FontWeight','bold');
ylabel('Best Fitness','FontWeight','bold');
title('GWO Convergence Curve','FontSize',12,'FontWeight','bold');
xlim([1, gwo_MaxIt]);

% ======================================================================
%  图3 - SSA 收敛曲线（春日青）
% ======================================================================
figure('Color','w','Name','SSA Convergence','Position',[50,150,560,360]);
plot(ssa_curve,'LineWidth',2,'Color',[0.42,0.78,0.60]);
grid on; box on;
xlabel('Iteration','FontWeight','bold');
ylabel('Best Fitness','FontWeight','bold');
title('SSA Convergence Curve','FontSize',12,'FontWeight','bold');
xlim([1, ssa_MaxIt]);

% ======================================================================
%  图4 - 三算法叠加对比地图
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

% --- PSO 固定分配线（红色底图） ---
if ~isempty(data.FID)
    for i = 1:size(data.FID,1)
        line([house_x(data.FID(i,1)), binan_x(data.FID(i,2))], ...
             [house_y(data.FID(i,1)), binan_y(data.FID(i,2))], ...
             'Color',[0.90,0.10,0.10,0.45],'LineWidth',0.8);
    end
end

% --- PSO 灵活分配线（红色，铺底） ---
for i = 1:nVar
    hIdx = data.DFenPei{i}(1);
    line([house_x(hIdx), binan_x(eID_PSO(i))], ...
         [house_y(hIdx), binan_y(eID_PSO(i))], ...
         'Color',[0.90,0.10,0.10,0.45],'LineWidth',0.8);
end

% --- GWO 差异线（紫色） ---
for i = find(diff_GWO)
    hIdx = data.DFenPei{i}(1);
    line([house_x(hIdx), binan_x(eID_GWO(i))], ...
         [house_y(hIdx), binan_y(eID_GWO(i))], ...
         'Color',[0.55,0.10,0.90,0.35],'LineWidth',1.0);
end

% --- SSA 差异线（春日青） ---
for i = find(diff_SSA)
    hIdx = data.DFenPei{i}(1);
    line([house_x(hIdx), binan_x(eID_SSA(i))], ...
         [house_y(hIdx), binan_y(eID_SSA(i))], ...
         'Color',[0.42,0.78,0.60,0.65],'LineWidth',1.0);
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
ax.FontSize = 11; ax.FontWeight = 'bold';
margin = 100;
xlim([min(house_x)-margin, max(house_x)+margin]);
ylim([min(house_y)-margin, max(house_y)+margin]);

title('PSO / GWO / SSA Allocation Comparison Map', ...
      'FontSize',14,'FontWeight','bold');
xlabel('X Coordinate Offset (m)','FontWeight','bold');
ylabel('Y Coordinate Offset (m)','FontWeight','bold');

% --- 图例（右上角） ---
h_pso_leg = line(NaN,NaN,'Color',[0.90,0.10,0.10,0.80],'LineWidth',2.0);
h_gwo_leg = line(NaN,NaN,'Color',[0.55,0.10,0.90,0.50],'LineWidth',2.0);
h_ssa_leg = line(NaN,NaN,'Color',[0.42,0.78,0.60,0.80],'LineWidth',2.0);

legend([h_shl, h_res, h_pso_leg, h_gwo_leg, h_ssa_leg], ...
    {'Shelter', ...
     'Residential', ...
     'PSO allocation', ...
     'GWO allocation (differs from PSO)', ...
     'SSA allocation (differs from PSO)'}, ...
    'Location','northeast','FontSize',7,'Box','on');

fprintf('\n全部完成！共耗时 %.1f 秒。\n', toc);
fprintf('\n图例说明:\n');
fprintf('  红色线   = PSO 全部分配（底图基准）\n');
fprintf('  紫色线   = GWO 与 PSO 不同的分配连线\n');
fprintf('  春日青线 = SSA 与 PSO 不同的分配连线\n');
fprintf('  绿色三角 = 避难场所\n');
fprintf('  蓝色圆点 = 住宅点位\n');

% ======================================================================
%  本地函数区
% ======================================================================
function score = shared_obj(x, S, w)
    x(x<0.01) = 0.01;
    X = ceil(x .* S.P);
    total_dist = S.alldis_fixed;
    Y = S.YFenPei_fixed;
    for i = 1:length(X)
        hID = S.DFenPei{i}(1);
        eID = S.DFenPei{i}(X(i)+1);
        total_dist = total_dist + S.dis(hID, eID);
        Y(eID) = Y(eID) + 12;
    end
    score = w.w1 * total_dist + w.w2 * var(Y);
end
