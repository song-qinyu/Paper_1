%% ============================================================
%  Hybrid Comparison Visualization  v2
%  Parallel Hybrid  (w_11.m)  vs  Pipeline Hybrid  (w_8.m)
%
%  新增完整指标计算：
%    领域KPI: TED, ATD, MID, SUR, BTV, CG, MET
%    算法KPI: Mean fitness, Std, NFE, NFE efficiency,
%             N_algos, Elite transfer, Diversity,
%             Parallelism, Stagnation escape,
%             Population ceiling, Multi-obj, Independent runs
%
%  运行方式：
%    先运行 w_8.m  (Pipeline) → 再运行 w_11.m (Parallel)
%    → 最后运行本文件
%
%  若工作区变量不存在，自动进入 demo 模式使用估算值
%% ============================================================
clc; close all; tic;

%% ========== 0. 颜色系统 ==========
C_GREEN = [88,  140,  90]/255;
C_GOLD  = [214, 164,  59]/255;
C_MID   = 0.5*C_GREEN + 0.5*C_GOLD;
C_LGR   = [160, 200, 130]/255;
C_LGO   = [240, 210, 140]/255;
C_W     = [252, 251, 248]/255;
C_GRID  = [0.88 0.88 0.85];

GRAD5 = {
    [88,  140,  90]/255,
    [138, 174,  90]/255,
    [178, 168,  80]/255,
    [204, 158,  60]/255,
    [214, 140,  45]/255
};

n_c = 256; h2 = n_c/2;
cmap_gg = [...
    linspace(C_GREEN(1),0.97,h2)', linspace(C_GREEN(2),0.97,h2)', linspace(C_GREEN(3),0.97,h2)';...
    linspace(0.97,C_GOLD(1),h2)',  linspace(0.97,C_GOLD(2),h2)',  linspace(0.97,C_GOLD(3),h2)'];

fprintf('==========================================================\n');
fprintf('  Hybrid Comparison Visualization  v2\n');
fprintf('  Parallel (w_11) vs Pipeline (w_8)\n');
fprintf('==========================================================\n\n');

%% ========== 1. Parallel 数据载入 & KPI计算 ==========
if exist('all_hub_fit','var') && exist('all_conv','var') && ...
   exist('DFenPei','var') && exist('data','var') && exist('all_hub_pos','var')
    fprintf('[INFO] Parallel: 使用工作区实际结果\n');

    % ── 基本统计 ──────────────────────────────────────────
    par_btv   = min(all_hub_fit);          % BTV: best fitness value
    par_mean  = mean(all_hub_fit);
    par_std   = std(all_hub_fit);
    par_worst = max(all_hub_fit);
    par_n_runs= N_RUNS;
    par_nfe_total = NFE_TOTAL;             % 75000

    % NFE efficiency: total fitness reduction per 1000 NFE
    par_init_fit = mean(all_conv(:,1));    % 初始适应度（均值）
    par_nfe_eff  = (par_init_fit - par_btv) / (par_nfe_total/1000);

    % ── 收敛曲线 ──────────────────────────────────────────
    par_conv_mean = mean(all_conv, 1);
    par_conv_std  = std(all_conv, 0, 1);
    n_sync        = size(all_conv, 2);
    par_nfe_axis  = round(linspace(NFE_TOTAL/n_sync, NFE_TOTAL, n_sync));

    % CG: 第一次达到最终值 95% 改进的 epoch（收敛代数）
    target_rel = 0.95 * (par_init_fit - par_btv);
    par_cg = find((par_init_fit - par_conv_mean) >= target_rel, 1);
    if isempty(par_cg), par_cg = n_sync; end
    par_cg_nfe = par_nfe_axis(par_cg);

    % ── 领域 KPI：从最优解计算 ────────────────────────────
    % 取最优运行的 best_pos
    [~, best_run] = min(all_hub_fit);
    X_par = all_hub_pos{best_run};

    Lb_p = ones(1, length(DFenPei));
    Ub_p = arrayfun(@(i) length(DFenPei{i})-1, 1:length(DFenPei));
    dim_p = length(DFenPei);

    % 固定分配
    alldis_fixed_p = 0;
    fixed_map_p = zeros(1, size(data.binan,1));
    if exist('B','var')
        for k = 1:length(B)
            if length(B{k})==1
                tb = B{k};
                alldis_fixed_p = alldis_fixed_p + data.dis(k, tb);
                fixed_map_p(tb) = fixed_map_p(tb) + 12;
            end
        end
    end

    % 每个住户的疏散距离
    all_dists_p = zeros(dim_p,1);
    shelter_load_p = zeros(size(data.binan,1),1);
    for i = 1:dim_p
        idx = max(Lb_p(i), min(round(X_par(i)), Ub_p(i)));
        sid = DFenPei{i}(idx+1);
        all_dists_p(i) = data.dis(DFenPei{i}(1), sid);
        shelter_load_p(sid) = shelter_load_p(sid) + 12;
    end

    par_btv_fit  = par_btv;                        % BTV = best fitness (变量部分)
    par_ted      = par_btv + alldis_fixed_p;        % TED
    total_pts_p  = size(data.start,1);
    par_atd      = par_ted / total_pts_p;           % ATD
    par_mid      = max(all_dists_p);                % MID
    % SUR: 被实际使用的避难所数量 / 总避难所数量
    used_shelters_p = shelter_load_p > 0;
    if exist('B','var')
        for k=1:length(B)
            if length(B{k})==1, used_shelters_p(B{k}) = true; end
        end
    end
    par_sur = sum(used_shelters_p) / size(data.binan,1) * 100;
    par_met = toc;    % 运行时间（本脚本执行时间，不含原始优化）

    % ── epoch win 统计 ───────────────────────────────────
    win_count = zeros(1,5);
    for r = 1:size(all_hub_fit,2)
        for ep = 1:n_sync
            ep_fits = squeeze(all_sync_hist(r,ep,:))';
            [~,w] = min(ep_fits); win_count(w) = win_count(w)+1;
        end
    end
    accept_rate   = squeeze(mean(all_accept,1));
    par_sync_mean = squeeze(mean(all_sync_hist,1));
    par_algo_names = ALGO_NAMES;

else
    fprintf('[WARN] Parallel: 未找到完整变量，使用 demo 估算值\n');
    par_btv   = 98483270;   par_mean  = 98619923;
    par_std   = 123515;     par_worst = 98817703;
    par_n_runs= 5;          par_nfe_total = 75000;
    par_nfe_eff = (99200000 - par_btv) / 75;
    par_ted   = par_btv + 0;      % demo: 无固定分配
    par_atd   = par_ted / 500;
    par_mid   = 3200;
    par_sur   = 68.5;
    par_met   = 9347;
    par_cg    = 14;   par_cg_nfe = 14*3750;
    n_sync    = 20;
    par_nfe_axis  = round(linspace(3750, 75000, n_sync));
    par_conv_mean = linspace(99200000, 98619923, n_sync);
    par_conv_std  = linspace(180000, 123515, n_sync);
    par_sync_mean = repmat(linspace(99200000,98619923,n_sync)',1,5) .* ...
                    (1 + 0.002*randn(n_sync,5));
    win_count     = [78, 11, 5, 4, 2];
    accept_rate   = repmat(linspace(80,60,n_sync)',1,5)/100;
    par_algo_names= {'NSGA','ABC','HLO','GWO','TOW'};
    par_btv_fit   = par_btv;
end

%% ========== 2. Pipeline 数据载入 & KPI计算 ==========
if exist('CC_all','var') && exist('stage_boundary','var') && ...
   exist('bestFit5','var') && exist('X_final','var')
    fprintf('[INFO] Pipeline: 使用工作区实际结果\n');

    pip_btv   = bestFit5;
    pip_cc    = CC_all;
    pip_sb    = stage_boundary;
    stage_starts_p = [1, pip_sb(1:4)+1];

    % 已由 w_8.m 计算的领域KPI
    pip_ted = TED;   pip_atd = ATD;
    pip_mid = MID;   pip_sur = SUR;
    pip_met = MET;

    % NFE
    pip_nfe_budgets = [20000,25000,25000,30000,50000];
    pip_nfe_total   = sum(pip_nfe_budgets);
    pip_init_fit    = pip_cc(1);
    pip_nfe_eff     = (pip_init_fit - pip_btv) / (pip_nfe_total/1000);

    % CG: Pipeline 阶段粒度（在哪个阶段结束时已达 95% 改进）
    target_pip = 0.95*(pip_init_fit - pip_btv);
    cumimpr_pip = max(pip_init_fit - pip_cc, 0);
    pip_cg_iter = find(cumimpr_pip >= target_pip, 1);
    if isempty(pip_cg_iter), pip_cg_iter = length(pip_cc); end
    pip_nfe_axis_ = linspace(0, pip_nfe_total, length(pip_cc));
    pip_cg_nfe    = pip_nfe_axis_(pip_cg_iter);
    % 换算成阶段号
    pip_cg_stage = 5;
    for s=1:5
        if pip_cg_iter <= pip_sb(s), pip_cg_stage=s; break; end
    end

    % per-stage improvement
    pip_imprs = zeros(1,5);
    for s=1:5
        seg = pip_cc(stage_starts_p(s):pip_sb(s));
        pip_imprs(s) = max(seg(1)-seg(end), 0);
    end

else
    fprintf('[WARN] Pipeline: 未找到完整变量，使用 demo 估算值\n');
    pip_btv   = 97800000;
    s1=linspace(100200000,99400000,250); s2=linspace(99400000,98900000,417);
    s3=linspace(98900000,98500000,417);  s4=linspace(98500000,98200000,375);
    s5=linspace(98200000,97800000,500);
    pip_cc=[s1,s2,s3,s4,s5]'; pip_sb=[250,667,1084,1459,1959];
    stage_starts_p=[1,pip_sb(1:4)+1];
    pip_ted=97800000; pip_atd=pip_ted/500; pip_mid=2800; pip_sur=72.5; pip_met=420;
    pip_nfe_budgets=[20000,25000,25000,30000,50000];
    pip_nfe_total=sum(pip_nfe_budgets);
    pip_init_fit=pip_cc(1);
    pip_nfe_eff=(pip_init_fit-pip_btv)/(pip_nfe_total/1000);
    pip_imprs=[800000,500000,100000,90000,4500];
    pip_cg_nfe=90000; pip_cg_stage=3;
end

pip_nfe_axis  = linspace(0, pip_nfe_total, length(pip_cc));
pip_stage_names= {'S1 ABC','S2 NSGA','S3 NSGA','S4 NSGA','S5 NSGA'};
pip_pop_sizes  = [40, 60, 60, 80, 100];
pip_mut_rates  = [NaN, 0.08, 0.15, 0.04, 0.02];

%% ========== 3. 全量指标汇总结构体 ==========
% ── 领域KPI ──────────────────────────────────────────────────
KPI.names_domain = {'TED (m)','ATD (m)','MID (m)','SUR (%)','BTV (m)','CG (NFE)','MET (s)'};
KPI.par_domain   = [par_ted, par_atd, par_mid, par_sur, par_btv, par_cg_nfe, par_met];
KPI.pip_domain   = [pip_ted, pip_atd, pip_mid, pip_sur, pip_btv, pip_cg_nfe, pip_met];
% 各指标越小越好 (1) 还是越大越好 (-1)
KPI.domain_dir   = [1, 1, 1, -1, 1, 1, 1];   % 1=lower better, -1=higher better

% ── 算法KPI ──────────────────────────────────────────────────
KPI.names_algo   = {'Mean Fitness (m)','Std Dev (m)','Total NFE','NFE Effic (m/kNFE)',...
                    'N Algorithms','Independent Runs','Pop Ceiling'};
KPI.par_algo     = [par_mean, par_std, par_nfe_total, par_nfe_eff, 5, par_n_runs, 50];
KPI.pip_algo     = [pip_btv,  0,       pip_nfe_total, pip_nfe_eff, 2, 1,          100];
KPI.algo_dir     = [1, 1, 1, -1, 0, -1, 0];  % 0=descriptive

% ── 定性属性 ─────────────────────────────────────────────────
KPI.names_qual   = {'Elite Transfer','Diversity Maint.','Parallelism',...
                    'Stagnation Escape','Multi-Obj Aware'};
KPI.par_qual     = {'Hub injection (worst-replace)','5 diverse algo types',...
                    'parfor (8 workers)','Hub refresh per epoch','NSGA only'};
KPI.pip_qual     = {'Top-K elite pool (K=30)','Mutation rate annealing',...
                    'Sequential stages','S3 high-mut breakout','All NSGA stages'};

% 控制台输出
fprintf('\n======= 完整 KPI 对比汇总 =======\n');
fprintf('%-22s  %14s  %14s\n','Metric','Parallel','Pipeline');
fprintf('%s\n',repmat('-',1,54));
for k=1:length(KPI.names_domain)
    fprintf('%-22s  %14.2f  %14.2f\n', KPI.names_domain{k}, KPI.par_domain(k), KPI.pip_domain(k));
end
fprintf('%s\n',repmat('-',1,54));
for k=1:length(KPI.names_algo)
    fprintf('%-22s  %14.2f  %14.2f\n', KPI.names_algo{k}, KPI.par_algo(k), KPI.pip_algo(k));
end
fprintf('%s\n',repmat('-',1,54));
for k=1:length(KPI.names_qual)
    fprintf('%-22s  %s\n         vs %s\n', KPI.names_qual{k}, KPI.par_qual{k}, KPI.pip_qual{k});
end
fprintf('=================================\n\n');

%% ========== Fig 1: 领域KPI 双向对比条形图 ==========
fig1 = figure('Color',C_W,'Name','comp_fig1_domain_kpi','Position',[30 50 1400 560]);

nDom = length(KPI.names_domain);
ax1 = axes('Color',C_W); hold on; box on; grid on;
ax1.GridColor=C_GRID; ax1.GridAlpha=.4; ax1.Color=C_W;

% 对每个指标归一化到 [0,1]，统一方向（1=更好）
dom_norm = zeros(2, nDom);
for k = 1:nDom
    vp = KPI.par_domain(k); vq = KPI.pip_domain(k);
    mx = max(abs(vp),abs(vq)); if mx<1e-10, mx=1; end
    if KPI.domain_dir(k)==1      % lower better
        dom_norm(1,k) = 1 - vp/mx;
        dom_norm(2,k) = 1 - vq/mx;
    elseif KPI.domain_dir(k)==-1 % higher better
        dom_norm(1,k) = vp/mx;
        dom_norm(2,k) = vq/mx;
    else
        dom_norm(1,k) = 0.5; dom_norm(2,k) = 0.5;
    end
end

bw1 = 0.35;
for k = 1:nDom
    % Parallel bar
    bpar = bar(k-bw1/2, dom_norm(1,k), bw1,'FaceColor',C_GREEN,'EdgeColor','none','FaceAlpha',.9);
    % Pipeline bar
    bpip = bar(k+bw1/2, dom_norm(2,k), bw1,'FaceColor',C_GOLD, 'EdgeColor','none','FaceAlpha',.9);

    % 原始数值标注
    vp = KPI.par_domain(k); vq = KPI.pip_domain(k);
    if vp > 1e5
        lp = sprintf('%.2fM',vp/1e6); lq = sprintf('%.2fM',vq/1e6);
    elseif vp > 1e3
        lp = sprintf('%.0f',vp);   lq = sprintf('%.0f',vq);
    else
        lp = sprintf('%.1f',vp);   lq = sprintf('%.1f',vq);
    end
    text(k-bw1/2, dom_norm(1,k)+0.03, lp,'HorizontalAlignment','center',...
        'FontSize',7.5,'Color',C_GREEN*0.7,'FontWeight','bold');
    text(k+bw1/2, dom_norm(2,k)+0.03, lq,'HorizontalAlignment','center',...
        'FontSize',7.5,'Color',C_GOLD*0.65,'FontWeight','bold');

    % 胜者标记
    if KPI.domain_dir(k) ~= 0
        if dom_norm(1,k) > dom_norm(2,k)+0.02
            text(k-bw1/2, dom_norm(1,k)+0.10,'WIN','HorizontalAlignment','center',...
                'FontSize',7,'Color',C_GREEN*0.6,'FontWeight','bold');
        elseif dom_norm(2,k) > dom_norm(1,k)+0.02
            text(k+bw1/2, dom_norm(2,k)+0.10,'WIN','HorizontalAlignment','center',...
                'FontSize',7,'Color',C_GOLD*0.6,'FontWeight','bold');
        end
    end
end

xticks(1:nDom); xticklabels(KPI.names_domain); xtickangle(20);
ylabel('Normalised Score  (higher = better)','FontSize',10,'FontWeight','bold');
title('Domain KPI Comparison  (TED / ATD / MID / SUR / BTV / CG / MET)', ...
    'FontSize',12,'FontWeight','bold','Color',[.15 .15 .15]);
p1=patch(NaN,NaN,C_GREEN,'EdgeColor','none');
p2=patch(NaN,NaN,C_GOLD,'EdgeColor','none');
legend([p1,p2],{'Parallel Hybrid','Pipeline Hybrid'},'Box','off','FontSize',10,'Location','northeast');
ylim([0 1.25]); set(ax1,'TickDir','out','FontSize',9,'Box','off');

exportgraphics(fig1,'comp_fig1_domain_kpi.png','Resolution',200);
fprintf('Saved: comp_fig1_domain_kpi.png\n');

%% ========== Fig 2: 算法KPI 对比条形图 ==========
fig2 = figure('Color',C_W,'Name','comp_fig2_algo_kpi','Position',[50 50 1400 540]);

nAlgo_kpi = length(KPI.names_algo);
ax2 = axes('Color',C_W); hold on; box on; grid on;
ax2.GridColor=C_GRID; ax2.GridAlpha=.4; ax2.Color=C_W;

algo_norm = zeros(2, nAlgo_kpi);
for k = 1:nAlgo_kpi
    vp = KPI.par_algo(k); vq = KPI.pip_algo(k);
    mx = max(abs(vp),abs(vq)); if mx<1e-10, mx=1; end
    if KPI.algo_dir(k)==1       % lower better
        algo_norm(1,k) = 1 - vp/mx; algo_norm(2,k) = 1 - vq/mx;
    elseif KPI.algo_dir(k)==-1  % higher better
        algo_norm(1,k) = vp/mx;     algo_norm(2,k) = vq/mx;
    else                         % descriptive
        algo_norm(1,k) = 0.5;       algo_norm(2,k) = 0.5;
    end
end

bw2 = 0.35;
for k = 1:nAlgo_kpi
    bar(k-bw2/2, algo_norm(1,k), bw2,'FaceColor',C_GREEN,'EdgeColor','none','FaceAlpha',.9);
    bar(k+bw2/2, algo_norm(2,k), bw2,'FaceColor',C_GOLD, 'EdgeColor','none','FaceAlpha',.9);

    vp=KPI.par_algo(k); vq=KPI.pip_algo(k);
    if abs(vp) > 1e5
        lp=sprintf('%.1fM',vp/1e6); lq=sprintf('%.1fM',vq/1e6);
    elseif abs(vp) > 999
        lp=sprintf('%.0f',vp); lq=sprintf('%.0f',vq);
    else
        lp=sprintf('%.1f',vp); lq=sprintf('%.1f',vq);
    end
    text(k-bw2/2, algo_norm(1,k)+0.03, lp,'HorizontalAlignment','center',...
        'FontSize',7.5,'Color',C_GREEN*0.7,'FontWeight','bold');
    text(k+bw2/2, algo_norm(2,k)+0.03, lq,'HorizontalAlignment','center',...
        'FontSize',7.5,'Color',C_GOLD*0.65,'FontWeight','bold');

    if KPI.algo_dir(k) ~= 0
        if algo_norm(1,k) > algo_norm(2,k)+0.02
            text(k-bw2/2, algo_norm(1,k)+0.10,'WIN','HorizontalAlignment','center',...
                'FontSize',7,'Color',C_GREEN*0.6,'FontWeight','bold');
        elseif algo_norm(2,k) > algo_norm(1,k)+0.02
            text(k+bw2/2, algo_norm(2,k)+0.10,'WIN','HorizontalAlignment','center',...
                'FontSize',7,'Color',C_GOLD*0.6,'FontWeight','bold');
        end
    else
        % 描述性指标在柱子内部显示实际值
        if algo_norm(1,k)==0.5 && algo_norm(2,k)==0.5
            text(k, 0.55,'(descriptive)','HorizontalAlignment','center',...
                'FontSize',6.5,'Color',[.5 .5 .5]);
        end
    end
end

xticks(1:nAlgo_kpi); xticklabels(KPI.names_algo); xtickangle(20);
ylabel('Normalised Score  (higher = better  |  grey = descriptive)','FontSize',10,'FontWeight','bold');
title('Algorithm KPI Comparison  (Mean / Std / NFE / Efficiency / N\_Algos / Runs / Pop)', ...
    'FontSize',12,'FontWeight','bold','Color',[.15 .15 .15]);
p1=patch(NaN,NaN,C_GREEN,'EdgeColor','none');
p2=patch(NaN,NaN,C_GOLD,'EdgeColor','none');
legend([p1,p2],{'Parallel Hybrid','Pipeline Hybrid'},'Box','off','FontSize',10,'Location','northeast');
ylim([0 1.25]); set(ax2,'TickDir','out','FontSize',9,'Box','off');

exportgraphics(fig2,'comp_fig2_algo_kpi.png','Resolution',200);
fprintf('Saved: comp_fig2_algo_kpi.png\n');

%% ========== Fig 3: 完整 Head-to-Head 表格图 ==========
fig3 = figure('Color',C_W,'Name','comp_fig3_full_table','Position',[70 50 1500 900]);
ax3 = axes('Position',[0 0 1 1],'Color',C_W,'XColor','none','YColor','none');
axis off; hold on;

% 表头
row_data = {
%  Metric                    Parallel                          Pipeline                   Winner(1=Par,-1=Pip,0=tie)
% ─── 领域KPI ───────────────────────────────────────────────────────────────────
'TED (Total Evacuation Dist)',  sprintf('%.0f m', par_ted),      sprintf('%.0f m', pip_ted),        sign(pip_ted - par_ted);
'ATD (Avg Travel Distance)',    sprintf('%.2f m', par_atd),      sprintf('%.2f m', pip_atd),        sign(pip_atd - par_atd);
'MID (Max Individual Dist)',    sprintf('%.0f m', par_mid),      sprintf('%.0f m', pip_mid),        sign(pip_mid - par_mid);
'SUR (Shelter Utilization %)',  sprintf('%.1f %%', par_sur),     sprintf('%.1f %%', pip_sur),       sign(par_sur - pip_sur);
'BTV (Best Fitness Value)',     sprintf('%.0f m', par_btv),      sprintf('%.0f m', pip_btv),        sign(pip_btv - par_btv);
'CG  (Conv Generation NFE)',    sprintf('%.0f NFE', par_cg_nfe), sprintf('%.0f NFE (S%d)', pip_cg_nfe, pip_cg_stage), sign(pip_cg_nfe - par_cg_nfe);
'MET (Mean Exec Time)',         sprintf('%.0f s', par_met),      sprintf('%.0f s', pip_met),        sign(pip_met - par_met);
% ─── 算法KPI（数值型）──────────────────────────────────────────────────────────
'Mean Fitness',                 sprintf('%.0f m', par_mean),     sprintf('N/A  (1 run)',0),          0;
'Std Deviation',                sprintf('%.0f m', par_std),      'N/A',                              0;
'Total NFE',                    sprintf('%d', par_nfe_total),    sprintf('%d', pip_nfe_total),        sign(pip_nfe_total - par_nfe_total);
'NFE Efficiency',               sprintf('%.0f m/kNFE', par_nfe_eff), sprintf('%.0f m/kNFE', pip_nfe_eff), sign(par_nfe_eff - pip_nfe_eff);
'Number of Algorithms',         '5 concurrent',                  '2 types (5 stages)',               0;
'Independent Runs',             sprintf('%d', par_n_runs),       '1',                                sign(par_n_runs - 1);
'Population Ceiling',           '50  (GWO)',                     '100  (S5 NSGA)',                   sign(100 - 50);
% ─── 定性属性 ──────────────────────────────────────────────────────────────────
'Elite Transfer',               'Hub injection (worst-replace)', 'Top-K elite pool (K=30)',          0;
'Diversity Maint.',             '5 diverse algo types',          'Mutation rate annealing',          0;
'Parallelism',                  'parfor  (8 workers)',            'Sequential stages',                1;
'Stagnation Escape',            'Hub refresh per epoch',         'S3 high-mut breakout',             0;
'Multi-Obj Aware',              'NSGA only',                     'All NSGA stages',                  -1;
};

nRow = size(row_data,1);
nCol = 4;   % Metric | Parallel | Pipeline | Winner

col_x  = [0.03, 0.30, 0.56, 0.84];
col_w  = [0.27, 0.26, 0.28, 0.14];
row_h  = 0.040;
top_y  = 0.94;

% 表头
hdr = {'Metric','Parallel Hybrid','Pipeline Hybrid','Winner'};
hdr_bg = [C_GREEN*0.6 + [1 1 1]*0.4];
for c = 1:nCol
    rectangle('Position',[col_x(c), top_y, col_w(c), row_h], ...
        'FaceColor',hdr_bg,'EdgeColor','w','LineWidth',0.5);
    text(col_x(c)+col_w(c)/2, top_y+row_h/2, hdr{c}, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontSize',9,'FontWeight','bold','Color',[0.1 0.1 0.1]);
end

% 分组标签位置（行索引）
group_rows = {1, 8, 15};
group_lbls = {'Domain KPIs', 'Algorithm KPIs (numeric)', 'Algorithm KPIs (qualitative)'};
group_clr  = {C_GREEN, C_MID, C_GOLD};

row_colors_par = {C_GREEN, C_GREEN, C_GREEN, C_GREEN, C_GREEN, C_GREEN, C_GREEN, ...
                  C_GREEN, C_GREEN, C_GREEN, C_GREEN, C_GREEN, C_GREEN, C_GREEN, ...
                  C_GREEN, C_GREEN, C_GREEN, C_GREEN, C_GREEN};

% 数据行
for r = 1:nRow
    y_pos = top_y - r*row_h;
    % 交替背景
    if mod(r,2)==0, bg=[0.96 0.97 0.96]; else, bg=C_W; end
    rectangle('Position',[col_x(1), y_pos, sum(col_w), row_h], ...
        'FaceColor',bg,'EdgeColor','w','LineWidth',0.3);

    metric_txt = row_data{r,1};
    par_txt    = row_data{r,2};
    pip_txt    = row_data{r,3};
    winner_dir = row_data{r,4};

    % 分组标签判断
    is_domain = (r <= 7);
    is_algo_n = (r >= 8 && r <= 14);
    is_qual   = (r >= 15);

    % metric名 — 分组用不同深度色
    if is_domain,  mc=[.1 .1 .1];
    elseif is_algo_n, mc=[.2 .2 .2];
    else, mc=[.3 .3 .3]; end

    text(col_x(1)+0.008, y_pos+row_h/2, metric_txt, ...
        'HorizontalAlignment','left','VerticalAlignment','middle', ...
        'FontSize',8.5,'FontWeight','bold','Color',mc);

    % Parallel值
    text(col_x(2)+col_w(2)/2, y_pos+row_h/2, par_txt, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontSize',8.5,'Color',[.15 .15 .15]);

    % Pipeline值
    text(col_x(3)+col_w(3)/2, y_pos+row_h/2, pip_txt, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontSize',8.5,'Color',[.15 .15 .15]);

    % Winner列
    if winner_dir == 1    % Parallel wins
        w_txt = 'Parallel'; w_clr = C_GREEN*0.65;
    elseif winner_dir == -1  % Pipeline wins
        w_txt = 'Pipeline'; w_clr = C_GOLD*0.65;
    else
        w_txt = '—'; w_clr = [0.55 0.55 0.55];
    end
    text(col_x(4)+col_w(4)/2, y_pos+row_h/2, w_txt, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontSize',8.5,'FontWeight','bold','Color',w_clr);

    % 分组分隔线
    if r==7 || r==14
        line([col_x(1), col_x(end)+col_w(end)], [y_pos, y_pos], ...
            'Color',C_MID,'LineWidth',1.2);
    end
end

% 分组标签（左侧竖条）
grp_ranges = {1:7, 8:14, 15:19};
for g=1:3
    rr = grp_ranges{g};
    y1 = top_y - rr(end)*row_h;
    y2 = top_y - (rr(1)-1)*row_h;
    rectangle('Position',[0.003, y1, 0.012, y2-y1], ...
        'FaceColor', group_clr{g}, 'EdgeColor','none');
    text(0.009, (y1+y2)/2, group_lbls{g}, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontSize',7,'Rotation',90,'Color','w','FontWeight','bold');
end

% 大标题
text(0.50, 0.985, 'Head-to-Head Full Metric Comparison — Parallel vs Pipeline Hybrid', ...
    'HorizontalAlignment','center','FontSize',13,'FontWeight','bold','Color',[.12 .12 .12]);

exportgraphics(fig3,'comp_fig3_full_table.png','Resolution',200);
fprintf('Saved: comp_fig3_full_table.png\n');

%% ========== Fig 4: 雷达图（全12维） ==========
fig4 = figure('Color',C_W,'Name','comp_fig4_radar_full','Position',[90 50 900 820]);

% 12维雷达（全量KPI归一化）
dims12 = {'TED','ATD','MID','SUR','BTV','CG',...
          'MET','Mean Fit','Std Dev','NFE Effic','Robustness','Multi-Obj'};
% Parallel得分（0-1，1=更优）
v_par12 = [
    1-par_ted/max(par_ted,pip_ted),            % TED lower better
    1-par_atd/max(par_atd,pip_atd),            % ATD lower better
    1-par_mid/max(par_mid,pip_mid),            % MID lower better
    par_sur/max(par_sur,pip_sur),              % SUR higher better
    1-par_btv/max(par_btv,pip_btv),           % BTV lower better
    1-par_cg_nfe/max(par_cg_nfe,pip_cg_nfe),  % CG lower better
    1-par_met/max(par_met,pip_met),            % MET lower better
    1-par_mean/max(par_mean,pip_btv*1.001),    % Mean lower better
    1-par_std/max(par_std,0.001),              % Std lower better
    par_nfe_eff/max(par_nfe_eff,pip_nfe_eff),  % Efficiency higher better
    par_n_runs/max(par_n_runs,1),              % Robustness: n_runs higher better
    0.40                                        % Multi-obj: NSGA only → lower
];
v_pip12 = [
    1-pip_ted/max(par_ted,pip_ted),
    1-pip_atd/max(par_atd,pip_atd),
    1-pip_mid/max(par_mid,pip_mid),
    pip_sur/max(par_sur,pip_sur),
    1-pip_btv/max(par_btv,pip_btv),
    1-pip_cg_nfe/max(par_cg_nfe,pip_cg_nfe),
    1-pip_met/max(par_met,pip_met),
    1-pip_btv/max(par_mean,pip_btv*1.001),
    0.99,                                       % Pipeline: 1 run → std = 0
    pip_nfe_eff/max(par_nfe_eff,pip_nfe_eff),
    1/max(par_n_runs,1),                        % 1 run → low robustness
    0.85                                        % All NSGA stages → high
];
v_par12 = max(0, min(1, v_par12));
v_pip12 = max(0, min(1, v_pip12));

nD12 = length(dims12);
theta12 = linspace(0, 2*pi, nD12+1); theta12(end) = [];

axes('Color',C_W,'Position',[0.08 0.05 0.84 0.88]); hold on;
for rv=[0.25,0.5,0.75,1.0]
    th_=linspace(0,2*pi,300);
    plot(cos(th_)*rv, sin(th_)*rv, '-', 'Color',C_GRID,'LineWidth',0.8);
    if rv < 1.0
        text(0, rv+0.04, sprintf('%.0f%%',rv*100), 'FontSize',7.5, ...
            'HorizontalAlignment','center','Color',[0.6 0.6 0.6]);
    end
end
for k=1:nD12
    plot([0,cos(theta12(k))],[0,sin(theta12(k))],'--','Color',C_GRID,'LineWidth',0.7);
    [xe,ye]=pol2cart(theta12(k),1.26);
    text(xe,ye,dims12{k},'HorizontalAlignment','center','FontSize',9,...
        'Color',[.2 .2 .2],'FontWeight','bold');
end

v_par12 = v_par12(:)'; v_pip12 = v_pip12(:)'; theta12 = theta12(:)';
vp12=[v_par12, v_par12(1)]; tp12=[theta12, theta12(1)];
[xp,yp]=pol2cart(tp12,vp12);
fill(xp,yp,C_GREEN,'FaceAlpha',.18,'EdgeColor','none');
hpar=plot(xp,yp,'-o','Color',C_GREEN,'LineWidth',2.5,'MarkerSize',6,...
    'MarkerFaceColor',C_GREEN,'MarkerEdgeColor','w');

vq12=[v_pip12, v_pip12(1)]; tq12=[theta12, theta12(1)];
[xq,yq]=pol2cart(tq12,vq12);
fill(xq,yq,C_GOLD,'FaceAlpha',.15,'EdgeColor','none');
hpip=plot(xq,yq,'--o','Color',C_GOLD,'LineWidth',2.5,'MarkerSize',6,...
    'MarkerFaceColor',C_GOLD,'MarkerEdgeColor','w');

% 数值标注
for k=1:nD12
    [xd,yd]=pol2cart(theta12(k), v_par12(k)*1.08);
    text(xd,yd,sprintf('%.2f',v_par12(k)),'FontSize',7,'Color',C_GREEN*0.7,...
        'HorizontalAlignment','center','FontWeight','bold');
    [xd2,yd2]=pol2cart(theta12(k), v_pip12(k)*0.88);
    text(xd2,yd2,sprintf('%.2f',v_pip12(k)),'FontSize',7,'Color',C_GOLD*0.7,...
        'HorizontalAlignment','center','FontWeight','bold');
end

legend([hpar,hpip],{'Parallel Hybrid','Pipeline Hybrid'}, ...
    'Location','southoutside','Orientation','horizontal','Box','off','FontSize',10);
axis equal off;
title('12-Dimension Full KPI Radar — Parallel vs Pipeline', ...
    'FontSize',13,'FontWeight','bold','Color',[.12 .12 .12],'Position',[0 1.38 0]);

exportgraphics(fig4,'comp_fig4_radar_full.png','Resolution',200);
fprintf('Saved: comp_fig4_radar_full.png\n');

%% ========== Fig 5: 收敛曲线 + CG标注 ==========
fig5 = figure('Color',C_W,'Name','comp_fig5_convergence','Position',[110 50 1300 560]);

ax5a = subplot(1,2,1); hold on; box on; grid on;
ax5a.GridColor=C_GRID; ax5a.GridAlpha=.45; ax5a.Color=C_W;

fill([par_nfe_axis, fliplr(par_nfe_axis)], ...
    [par_conv_mean+par_conv_std, fliplr(par_conv_mean-par_conv_std)], ...
    C_GREEN,'FaceAlpha',.15,'EdgeColor','none');
plot(par_nfe_axis, par_conv_mean,'-o','Color',C_GREEN,'LineWidth',2.5,...
    'MarkerSize',4,'MarkerFaceColor',C_GREEN,'MarkerEdgeColor','w',...
    'DisplayName',sprintf('Parallel Hub (mean, n=%d)',par_n_runs));

pip_nfe_axis = linspace(0, pip_nfe_total, length(pip_cc));
stage_starts_ = [1, pip_sb(1:4)+1];
for s=1:5
    sx = stage_starts_(s):pip_sb(s);
    plot(pip_nfe_axis(sx), pip_cc(sx),'-','Color',GRAD5{s},'LineWidth',2.5,...
        'DisplayName',sprintf('Pipeline %s',pip_stage_names{s}));
end
for s=1:4
    xb=pip_nfe_axis(pip_sb(s));
    xline(xb,'--','Color',[0.65 0.65 0.6],'LineWidth',1,'HandleVisibility','off');
end

% CG 标注
plot(par_cg_nfe, par_conv_mean(par_cg),'v','MarkerSize',12,...
    'MarkerFaceColor',C_GREEN,'MarkerEdgeColor','w','LineWidth',0.8,...
    'DisplayName',sprintf('Par CG @ %dkNFE',round(par_cg_nfe/1000)));
plot(pip_cg_nfe, pip_cc(min(round(pip_cg_nfe/pip_nfe_total*length(pip_cc)),length(pip_cc))),...
    'v','MarkerSize',12,'MarkerFaceColor',C_GOLD,'MarkerEdgeColor','w','LineWidth',0.8,...
    'DisplayName',sprintf('Pip CG @ S%d (%.0fkNFE)',pip_cg_stage,pip_cg_nfe/1000));

plot(par_nfe_axis(end), par_btv,'p','MarkerSize',13,...
    'MarkerFaceColor',C_GREEN,'MarkerEdgeColor','w','LineWidth',0.8,...
    'DisplayName',sprintf('Par BTV: %.0fM',par_btv/1e6));
plot(pip_nfe_axis(end), pip_btv,'p','MarkerSize',13,...
    'MarkerFaceColor',C_GOLD,'MarkerEdgeColor','w','LineWidth',0.8,...
    'DisplayName',sprintf('Pip BTV: %.0fM',pip_btv/1e6));

xlabel('NFE','FontSize',10,'FontWeight','bold');
ylabel('Best Fitness (m)','FontSize',10,'FontWeight','bold');
title('Convergence + CG (▽) + BTV (★)','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
legend('Location','northeast','Box','off','FontSize',8);
set(ax5a,'TickDir','out','FontSize',9,'Box','off');

% 右：归一化相对收敛速率
ax5b = subplot(1,2,2); hold on; box on; grid on;
ax5b.GridColor=C_GRID; ax5b.GridAlpha=.45; ax5b.Color=C_W;

par_init = par_conv_mean(1);
pip_init = pip_cc(1);
par_rel  = (par_init - par_conv_mean)/par_init*100;
pip_rel  = max(pip_init - pip_cc, 0)/pip_init*100;

fill([par_nfe_axis, fliplr(par_nfe_axis)], ...
    [par_rel + par_conv_std/par_init*100, fliplr(max(par_rel-par_conv_std/par_init*100,0))], ...
    C_GREEN,'FaceAlpha',.15,'EdgeColor','none');
plot(par_nfe_axis, par_rel,'-o','Color',C_GREEN,'LineWidth',2.5,...
    'MarkerSize',4,'MarkerFaceColor',C_GREEN,'MarkerEdgeColor','w','DisplayName','Parallel');
plot(pip_nfe_axis, pip_rel,'--','Color',C_GOLD,'LineWidth',2.5,'DisplayName','Pipeline');

% 95% 线
yline(95,'--','Color',[.5 .5 .5],'LineWidth',1.2,'Label','95% Conv');
plot(par_cg_nfe, 95,'v','MarkerSize',10,'MarkerFaceColor',C_GREEN,...
    'MarkerEdgeColor','w','HandleVisibility','off');
if pip_cg_nfe <= pip_nfe_total
    plot(pip_cg_nfe, 95,'v','MarkerSize',10,'MarkerFaceColor',C_GOLD,...
        'MarkerEdgeColor','w','HandleVisibility','off');
end

xlabel('NFE','FontSize',10,'FontWeight','bold');
ylabel('Relative Improvement (%)','FontSize',10,'FontWeight','bold');
title('Normalised Convergence Rate','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
legend('Location','southeast','Box','off','FontSize',9);
set(ax5b,'TickDir','out','FontSize',9,'Box','off');

sgtitle('Convergence Comparison — BTV & CG Marked','FontSize',13,'FontWeight','bold','Color',[.12 .12 .12]);
exportgraphics(fig5,'comp_fig5_convergence.png','Resolution',200);
fprintf('Saved: comp_fig5_convergence.png\n');

%% ========== Fig 6: 疏散领域KPI分组柱状图 ==========
fig6 = figure('Color',C_W,'Name','comp_fig6_evacuation_kpi','Position',[130 50 1300 520]);

% 6A: TED + ATD + MID
ax6a = subplot(1,3,1); hold on; box on; grid on;
ax6a.GridColor=C_GRID; ax6a.GridAlpha=.45; ax6a.Color=C_W;
dist_kpis    = [par_ted/1e6, par_atd, par_mid; pip_ted/1e6, pip_atd, pip_mid];
dist_names   = {'TED (M m)','ATD (m)','MID (m)'};
dist_scale   = [1e6, 1, 1];
bw6=0.35;
for k=1:3
    bar(k-bw6/2, dist_kpis(1,k), bw6,'FaceColor',C_GREEN,'EdgeColor','none','FaceAlpha',.9);
    bar(k+bw6/2, dist_kpis(2,k), bw6,'FaceColor',C_GOLD, 'EdgeColor','none','FaceAlpha',.9);
    text(k-bw6/2, dist_kpis(1,k)*1.04, sprintf('%.2f',dist_kpis(1,k)),...
        'HorizontalAlignment','center','FontSize',8,'Color',C_GREEN*0.7,'FontWeight','bold');
    text(k+bw6/2, dist_kpis(2,k)*1.04, sprintf('%.2f',dist_kpis(2,k)),...
        'HorizontalAlignment','center','FontSize',8,'Color',C_GOLD*0.65,'FontWeight','bold');
end
xticks(1:3); xticklabels(dist_names);
ylabel('Distance','FontSize',10,'FontWeight','bold');
title('Distance Metrics','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
p1=patch(NaN,NaN,C_GREEN,'EdgeColor','none'); p2=patch(NaN,NaN,C_GOLD,'EdgeColor','none');
legend([p1,p2],{'Parallel','Pipeline'},'Box','off','FontSize',9);
set(ax6a,'TickDir','out','FontSize',9,'Box','off');

% 6B: SUR
ax6b = subplot(1,3,2); hold on; box on; grid on;
ax6b.GridColor=C_GRID; ax6b.GridAlpha=.45; ax6b.Color=C_W;
bar(1, par_sur, 0.55,'FaceColor',C_GREEN,'EdgeColor','none','FaceAlpha',.9);
bar(2, pip_sur, 0.55,'FaceColor',C_GOLD, 'EdgeColor','none','FaceAlpha',.9);
text(1, par_sur+1, sprintf('%.1f%%',par_sur),'HorizontalAlignment','center','FontSize',11,...
    'Color',C_GREEN*0.7,'FontWeight','bold');
text(2, pip_sur+1, sprintf('%.1f%%',pip_sur),'HorizontalAlignment','center','FontSize',11,...
    'Color',C_GOLD*0.65,'FontWeight','bold');
xticks(1:2); xticklabels({'Parallel','Pipeline'});
ylabel('Shelter Utilization Rate (%)','FontSize',10,'FontWeight','bold');
title('SUR — Resource Allocation','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
ylim([0, max(par_sur,pip_sur)*1.25]);
set(ax6b,'TickDir','out','FontSize',9,'Box','off');

% 6C: MET
ax6c = subplot(1,3,3); hold on; box on; grid on;
ax6c.GridColor=C_GRID; ax6c.GridAlpha=.45; ax6c.Color=C_W;
bar(1, par_met, 0.55,'FaceColor',C_GREEN,'EdgeColor','none','FaceAlpha',.9);
bar(2, pip_met, 0.55,'FaceColor',C_GOLD, 'EdgeColor','none','FaceAlpha',.9);
text(1, par_met+par_met*0.04, sprintf('%.0f s',par_met),'HorizontalAlignment','center','FontSize',11,...
    'Color',C_GREEN*0.7,'FontWeight','bold');
text(2, pip_met+pip_met*0.04, sprintf('%.0f s',pip_met),'HorizontalAlignment','center','FontSize',11,...
    'Color',C_GOLD*0.65,'FontWeight','bold');
xticks(1:2); xticklabels({'Parallel','Pipeline'});
ylabel('Execution Time (s)','FontSize',10,'FontWeight','bold');
title('MET — Computational Overhead','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
set(ax6c,'TickDir','out','FontSize',9,'Box','off');

sgtitle('Evacuation Domain KPIs — TED / ATD / MID / SUR / MET','FontSize',13,'FontWeight','bold','Color',[.12 .12 .12]);
exportgraphics(fig6,'comp_fig6_evacuation_kpi.png','Resolution',200);
fprintf('Saved: comp_fig6_evacuation_kpi.png\n');

%% ========== Fig 7: 原有的 7 张图（保留全部）==========
%  （直接复用原 hybrid_comparison_viz.m 的 Fig1~Fig7，
%    新增数据已注入工作区，其余图不变）

% Fig7A: NFE预算分配
fig7 = figure('Color',C_W,'Name','comp_fig7_nfe_budget','Position',[150 50 1200 520]);
ax7a = subplot(1,2,1); hold on; box on; grid on;
ax7a.GridColor=C_GRID; ax7a.GridAlpha=.45; ax7a.Color=C_W;
epoch_nfe_ = par_nfe_axis(2)-par_nfe_axis(1);
bh7a = bar(1:n_sync, repmat(epoch_nfe_,1,n_sync),'FaceColor','flat','EdgeColor','none','BarWidth',0.80);
for ep=1:n_sync
    t=(ep-1)/(n_sync-1); bh7a.CData(ep,:)=(1-t)*C_GREEN+t*C_GOLD;
end
xlabel('Sync Epoch','FontSize',10,'FontWeight','bold');
ylabel('NFE per Epoch','FontSize',10,'FontWeight','bold');
title(sprintf('Parallel — Uniform NFE/Epoch = %d',epoch_nfe_),'FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
set(ax7a,'TickDir','out','FontSize',9,'Box','off'); ylim([0 epoch_nfe_*1.3]);

ax7b = subplot(1,2,2); hold on; box on; grid on;
ax7b.GridColor=C_GRID; ax7b.GridAlpha=.45; ax7b.Color=C_W;
bh7b = bar(pip_nfe_budgets,'FaceColor','flat','EdgeColor','none','BarWidth',0.68);
for s=1:5, bh7b.CData(s,:)=GRAD5{s}; end
pip_total_=sum(pip_nfe_budgets);
for s=1:5
    pct=pip_nfe_budgets(s)/pip_total_*100;
    text(s, pip_nfe_budgets(s)+pip_total_*0.012,...
        sprintf('%.0f%%\n(%dk)',pct,pip_nfe_budgets(s)/1000),...
        'HorizontalAlignment','center','FontSize',9,'Color',[.2 .2 .2]);
end
xticks(1:5); xticklabels(pip_stage_names);
ylabel('NFE Budget','FontSize',10,'FontWeight','bold');
title(sprintf('Pipeline — Staged NFE (total=%dk)',pip_total_/1000),'FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
set(ax7b,'TickDir','out','FontSize',9,'Box','off');
sgtitle('NFE Budget Allocation','FontSize',13,'FontWeight','bold','Color',[.12 .12 .12]);
exportgraphics(fig7,'comp_fig7_nfe_budget.png','Resolution',200);
fprintf('Saved: comp_fig7_nfe_budget.png\n');

%% ========== 输出汇总 ==========
fprintf('\n==========================================================\n');
fprintf('  对比可视化 v2 完成！共保存 7 张图\n');
fprintf('  comp_fig1_domain_kpi.png    — 领域KPI归一化条形对比\n');
fprintf('  comp_fig2_algo_kpi.png      — 算法KPI归一化条形对比\n');
fprintf('  comp_fig3_full_table.png    — 完整 Head-to-Head 表格\n');
fprintf('  comp_fig4_radar_full.png    — 12维全量KPI雷达图\n');
fprintf('  comp_fig5_convergence.png   — 收敛曲线 + BTV & CG标注\n');
fprintf('  comp_fig6_evacuation_kpi.png— 疏散领域KPI分组图\n');
fprintf('  comp_fig7_nfe_budget.png    — NFE预算分配\n');
fprintf('==========================================================\n');
fprintf('  [Parallel] BTV=%.0f  ATD=%.2f  MID=%.0f  SUR=%.1f%%  CG=%dkNFE\n',...
    par_btv, par_atd, par_mid, par_sur, round(par_cg_nfe/1000));
fprintf('  [Pipeline] BTV=%.0f  ATD=%.2f  MID=%.0f  SUR=%.1f%%  CG=S%d\n',...
    pip_btv, pip_atd, pip_mid, pip_sur, pip_cg_stage);
fprintf('==========================================================\n');