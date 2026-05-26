%% ============================================================
%  Parallel Integrative Hybrid: DE ‖ GWO ‖ CS ‖ TOW ‖ PO
%  疏散分配优化 - 五算法并行协同混合框架
%% ============================================================
clc; clear; close all; tic;
fprintf('============================================================\n');
fprintf('  Parallel Integrative Hybrid: DE ‖ GWO ‖ CS ‖ TOW ‖ PO\n');
fprintf('  五算法并行协同 | 同步信息交换 | 中央精英池\n');
fprintf('============================================================\n\n');

%% ======================== 0. 数据加载与预处理 ========================
if exist('sj5.mat','file'), load('sj5.mat');
else, error('未找到 sj5.mat'); end

if exist('dis','var'), data.dis = dis; end
raw_x = data.start(:,1); raw_y = data.start(:,2);
offset_x = min(raw_x); offset_y = min(raw_y);
house_x = raw_x - offset_x;  house_y = raw_y - offset_y;
binan_x = data.binan(:,1) - offset_x;  binan_y = data.binan(:,2) - offset_y;

dim = length(DFenPei);
Lb  = ones(1, dim);
Ub  = arrayfun(@(i) length(DFenPei{i})-1, 1:dim);

FID = []; alldis_fixed = 0;
YFP = zeros(1, size(data.binan,1));
for k = 1:length(B)
    if length(B{k}) == 1
        tb = B{k};
        FID = [FID; k, tb];
        alldis_fixed = alldis_fixed + data.dis(k, tb);
        YFP(tb) = YFP(tb) + 12;
    end
end

fobj = @(x) par_fobj(x, DFenPei, data.dis, Lb, Ub);

%% ======================== 1. 全局参数 ========================
n_DE  = 30; n_GWO = 30; n_CS  = 20; n_TOW = 20; n_PO  = 60;
N_total = n_DE + n_GWO + n_CS + n_TOW + n_PO;

% ---- 与 Pipeline 对齐迭代步数 ----
% Pipeline 总预算：20000+25000+25000+30000+50000 = 150000 NFE
% Parallel 每轮评价次数 = N_total = 160 次
% 因此轮数 = 150000 / 160 ≈ 937，取 937
MaxSyncIter = floor(150000 / N_total);

K_elite   = 20;
sync_freq = 5;

% 算子特定参数
F_DE = 0.7; CR_DE = 0.8;
pa_CS = 0.20; beta_CS = 1.5;
alpha_tow = 0.995; sigma_init = 4.0;
n_parties = 3; n_members = 20;
a_init = 2.0;

% 可视化颜色（单色，Parallel 不分阶段）
par_color = [0.15 0.45 0.85];

%% ======================== 2. 种群初始化 ========================
fprintf('[初始化] 生成各子算法种群...\n');
init_pop_fn = @(n, d, L, U) round(repmat(L,n,1) + rand(n,d).*repmat(U-L,n,1));

pop_DE  = init_pop_fn(n_DE,  dim, Lb, Ub);
pop_GWO = init_pop_fn(n_GWO, dim, Lb, Ub);
pop_CS  = init_pop_fn(n_CS,  dim, Lb, Ub);
pop_TOW = init_pop_fn(n_TOW, dim, Lb, Ub);
pop_PO  = init_pop_fn(n_PO,  dim, Lb, Ub);

fit_DE  = arrayfun(@(i) fobj(pop_DE(i,:)),  (1:n_DE)');
fit_GWO = arrayfun(@(i) fobj(pop_GWO(i,:)), (1:n_GWO)');
fit_CS  = arrayfun(@(i) fobj(pop_CS(i,:)),  (1:n_CS)');
fit_TOW = arrayfun(@(i) fobj(pop_TOW(i,:)), (1:n_TOW)');
fit_PO  = arrayfun(@(i) fobj(pop_PO(i,:)),  (1:n_PO)');

all_pop = [pop_DE; pop_GWO; pop_CS; pop_TOW; pop_PO];
all_fit = [fit_DE; fit_GWO; fit_CS; fit_TOW; fit_PO];
[~, si] = sort(all_fit);
elite_pool = all_pop(si(1:K_elite), :);
elite_fit  = all_fit(si(1:K_elite));

global_best_pos = elite_pool(1,:);
global_best_fit = elite_fit(1);

[~,gs] = sort(fit_GWO);
Alpha_pos = pop_GWO(gs(1),:); Alpha_score = fit_GWO(gs(1));
Beta_pos  = pop_GWO(gs(2),:); Beta_score  = fit_GWO(gs(2));
Delta_pos = pop_GWO(gs(3),:); Delta_score = fit_GWO(gs(3));

CC_global = zeros(MaxSyncIter, 1);
fprintf('  种群总规模: %d | 精英池: %d | 并行轮数: %d\n\n', N_total, K_elite, MaxSyncIter);

%% ======================== 3. 并行主循环 ========================
for iter = 1:MaxSyncIter
    % ---- DE ----
    for i = 1:n_DE
        r = randperm(n_DE, 3);
        v = round(pop_DE(i,:) + F_DE*(global_best_pos - pop_DE(i,:)) + ...
                  F_DE*(pop_DE(r(1),:) - pop_DE(r(2),:)));
        v = max(Lb, min(Ub, v));
        mask = rand(1,dim) < CR_DE; mask(randi(dim)) = true;
        trial = pop_DE(i,:); trial(mask) = v(mask);
        ft = fobj(trial);
        if ft < fit_DE(i), pop_DE(i,:) = trial; fit_DE(i) = ft; end
    end
    [best_DE_fit, bi] = min(fit_DE); best_DE_pos = pop_DE(bi,:);

    % ---- GWO ----
    a_gwo = a_init - iter*(a_init/MaxSyncIter);
    for i = 1:n_GWO
        for j = 1:dim
            X1 = Alpha_pos(j) - (2*a_gwo*rand()-a_gwo)*abs(2*rand()*Alpha_pos(j) - pop_GWO(i,j));
            X2 = Beta_pos(j)  - (2*a_gwo*rand()-a_gwo)*abs(2*rand()*Beta_pos(j)  - pop_GWO(i,j));
            X3 = Delta_pos(j) - (2*a_gwo*rand()-a_gwo)*abs(2*rand()*Delta_pos(j) - pop_GWO(i,j));
            pop_GWO(i,j) = round((X1+X2+X3)/3);
        end
        pop_GWO(i,:) = max(Lb, min(Ub, pop_GWO(i,:)));
        fit_GWO(i) = fobj(pop_GWO(i,:));
        if fit_GWO(i) < Alpha_score
            Delta_score=Beta_score; Delta_pos=Beta_pos;
            Beta_score=Alpha_score; Beta_pos=Alpha_pos;
            Alpha_score=fit_GWO(i); Alpha_pos=pop_GWO(i,:);
        end
    end

    % ---- CS ----
    for i = 1:n_CS
        levy = par_levy(beta_CS, dim);
        new_nest = round(pop_CS(i,:) + 0.01*levy.*(pop_CS(i,:) - global_best_pos));
        new_nest = max(Lb, min(Ub, new_nest));
        ft = fobj(new_nest);
        if ft < fit_CS(i), pop_CS(i,:) = new_nest; fit_CS(i) = ft; end
    end
    [best_CS_fit, bi] = min(fit_CS); best_CS_pos = pop_CS(bi,:);

    % ---- TOW ----
    sigma_cur = sigma_init * alpha_tow^iter;
    for i = 1:n_TOW
        step = round(0.5*(global_best_pos - pop_TOW(i,:)) + sigma_cur*randn(1,dim));
        pop_TOW(i,:) = max(Lb, min(Ub, pop_TOW(i,:) + step));
        fit_TOW(i) = fobj(pop_TOW(i,:));
    end
    [best_TOW_fit, bi] = min(fit_TOW); best_TOW_pos = pop_TOW(bi,:);

    % ---- PO ----
    for p = 1:n_parties
        idx_p = (p-1)*n_members+1 : p*n_members;
        [~,ll] = min(fit_PO(idx_p)); ldr_pos = pop_PO(idx_p(ll),:);
        for m = idx_p
            pop_PO(m,:) = max(Lb, min(Ub, round(pop_PO(m,:) + rand()*(ldr_pos - pop_PO(m,:)))));
            fit_PO(m) = fobj(pop_PO(m,:));
        end
    end
    [best_PO_fit, bi] = min(fit_PO); best_PO_pos = pop_PO(bi,:);

    % ---- 同步信息交换 ----
    sub_fits = [best_DE_fit, Alpha_score, best_CS_fit, best_TOW_fit, best_PO_fit];
    sub_pos  = [best_DE_pos; Alpha_pos; best_CS_pos; best_TOW_pos; best_PO_pos];
    [cur_best_fit, cb_idx] = min(sub_fits);
    if cur_best_fit < global_best_fit
        global_best_fit = cur_best_fit;
        global_best_pos = sub_pos(cb_idx,:);
    end

    if mod(iter, sync_freq) == 0
        all_pop_cur = [pop_DE; pop_GWO; pop_CS; pop_TOW; pop_PO];
        all_fit_cur = [fit_DE; fit_GWO; fit_CS; fit_TOW; fit_PO];
        [~, si] = sort(all_fit_cur);
        elite_pool = all_pop_cur(si(1:K_elite), :);
        elite_fit  = all_fit_cur(si(1:K_elite));
        [~, w] = max(fit_DE);  pop_DE(w,:)  = elite_pool(randi(K_elite),:); fit_DE(w)  = fobj(pop_DE(w,:));
        [~, w] = max(fit_GWO); pop_GWO(w,:) = elite_pool(randi(K_elite),:); fit_GWO(w) = fobj(pop_GWO(w,:));
    end

    CC_global(iter) = global_best_fit;
    if mod(iter, 100) == 0
        fprintf('  Iter %4d / %d | Global Best: %.2f m\n', iter, MaxSyncIter, global_best_fit);
    end
end

X_final = global_best_pos;

%% ======================== 4. 8项评价指标（与 Pipeline 统一格式）========================
TED = global_best_fit + alldis_fixed;
total_pts = size(data.start, 1);
ATD = TED / total_pts;

all_dists = zeros(total_pts, 1);
for i = 1:dim
    idx = max(Lb(i), min(X_final(i), Ub(i)));
    all_dists(i) = data.dis(DFenPei{i}(1), DFenPei{i}(idx+1));
end
for i = 1:size(FID,1)
    all_dists(dim+i) = data.dis(FID(i,1), FID(i,2));
end
MID = max(all_dists);

dyn_bins = arrayfun(@(i) DFenPei{i}(min(X_final(i)+1, length(DFenPei{i}))), 1:dim);
SUR = length(unique([dyn_bins, FID(:,2)'])) / size(data.binan,1) * 100;
MET = toc;

fprintf('\n  [8项评价指标]\n');
fprintf('  TED (总疏散距离):   %.2f m\n', TED);
fprintf('  ATD (平均疏散距离): %.2f m\n', ATD);
fprintf('  MID (最大单点距离): %.2f m\n', MID);
fprintf('  SUR (避难所利用率): %.2f %%\n', SUR);
fprintf('  BTV (最终适应度):   %.4f\n',   global_best_fit);
fprintf('  MET (总运行时间):   %.2f s\n', MET);
fprintf('  总迭代步数:         %d\n',     MaxSyncIter);
fprintf('========================================================\n');

%% ======================== 5. 可视化（与 Pipeline 统一格式）========================

% --- 图1：收敛曲线 ---
figure('Name','Parallel Integrative 收敛','Color','w','Position',[50,50,980,460]);
hold on; box on; grid on;
plot(1:MaxSyncIter, CC_global, 'Color', par_color, 'LineWidth', 2.5, ...
     'DisplayName','Parallel: DE‖GWO‖CS‖TOW‖PO');
xlabel('Cumulative Iteration', 'FontWeight','bold');
ylabel('Best Fitness — Total Distance (m)', 'FontWeight','bold');
title('Parallel Integrative Hybrid (DE‖GWO‖CS‖TOW‖PO)', 'FontSize',13,'FontWeight','bold');
legend('Location','northeast');

% --- 图2：各子算法最终贡献柱状图 ---
sub_names = {'DE','GWO','CS','TOW','PO'};
sub_colors = {[0.20 0.60 1.0],[0.10 0.75 0.30],[0.95 0.55 0.0],[0.80 0.20 0.20],[0.55 0.10 0.85]};
sub_bests  = [min(fit_DE), Alpha_score, min(fit_CS), min(fit_TOW), min(fit_PO)];

figure('Name','各子算法贡献','Color','w','Position',[200,200,680,430]);
b = bar(sub_bests, 'FaceColor','flat');
for s=1:5, b.CData(s,:) = sub_colors{s}; end
set(gca,'XTickLabel', sub_names, 'FontSize',12);
xlabel('Sub-Algorithm', 'FontWeight','bold');
ylabel('Final Best Fitness (m)', 'FontWeight','bold');
title('Sub-Algorithm Final Performance — Parallel Integrative','FontSize',12,'FontWeight','bold');
grid on; box on;
for s=1:5
    text(s, sub_bests(s)+max(sub_bests)*0.005, sprintf('%.0f', sub_bests(s)), ...
        'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
end

% --- 图3：2D 疏散分配地图 ---
figure('Color','w','Name','Parallel Integrative Final Map','Position',[100,100,900,800]);
hold on; box on;
if isfield(data,'road')
    for i=1:length(data.road)
        plot(data.road{i}(:,1)-offset_x, data.road{i}(:,2)-offset_y, ...
             'Color',[0.88 0.88 0.88],'LineWidth',0.5);
    end
end
for i=1:dim
    oi = max(1, min(X_final(i), length(DFenPei{i})-1));
    line([house_x(DFenPei{i}(1)), binan_x(DFenPei{i}(oi+1))], ...
         [house_y(DFenPei{i}(1)), binan_y(DFenPei{i}(oi+1))], ...
         'Color',[1 0 0 0.12],'LineWidth',0.35);
end
for i=1:size(FID,1)
    line([house_x(FID(i,1)), binan_x(FID(i,2))], ...
         [house_y(FID(i,1)), binan_y(FID(i,2))], ...
         'Color',[1 0 0 0.12],'LineWidth',0.35);
end
h_res = scatter(house_x, house_y, 10, [0 0.2 0.6], 'filled');
h_shl = scatter(binan_x, binan_y, 75, 'g','^','filled','MarkerEdgeColor','k');
axis equal; axis tight; grid on;
ax=gca; ax.XAxis.Exponent=0; ax.YAxis.Exponent=0;
title('Parallel Integrative Optimized Allocation Map','FontSize',13,'FontWeight','bold');
legend([h_shl,h_res],{'Shelter','Residential'},'Location','northeast');

fprintf('\n[完成] Parallel Integrative 运行结束。\n');

%% ======================== 子函数 ========================
function fitness = par_fobj(x, DFenPei, dis_mat, Lb, Ub)
    fitness = 0;
    for i = 1:length(DFenPei)
        idx = max(Lb(i), min(round(x(i)), Ub(i)));
        fitness = fitness + dis_mat(DFenPei{i}(1), DFenPei{i}(idx+1));
    end
end

function levy = par_levy(beta, dim)
    sigma = (gamma(1+beta)*sin(pi*beta/2) / ...
             (gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    levy = randn(1,dim)*sigma ./ (abs(randn(1,dim)).^(1/beta));
end