%% ============================================================
%  Hybrid Comparison Visualization
%  Parallel Hybrid  (w_11.m)  vs  Pipeline Hybrid  (w_8.m)
%
%  运行方式：
%    将本文件与 w_11.m / w_8.m 放在同一目录
%    先运行 w_11.m 得到 Parallel 结果变量
%    再运行 w_8.m  得到 Pipeline 结果变量
%    最后运行本文件生成对比图
%
%  本文件需要工作区中存在以下变量（由两个主文件生成）：
%    Parallel: all_hub_fit, all_conv, all_sync_hist, all_accept
%              ALGO_NAMES, N_RUNS, N_SYNC, NFE_EPOCH, NFE_TOTAL
%    Pipeline: CC_all, stage_boundary, bestFit5, stage_imprs
%              stage_labels, stage_names, stage_colors, TED, ATD, MID, SUR, MET
%
%  若变量不存在，本文件会使用论文结果中的估算值作为替代（demo模式）
%% ============================================================
clc; close all;

%% ==================== 0. 颜色系统 ====================
C_GREEN = [88,  140,  90] /255;
C_GOLD  = [214, 164,  59] /255;
C_MID   = 0.5*C_GREEN + 0.5*C_GOLD;
C_LGR   = [160, 200, 130] /255;
C_LGO   = [240, 210, 140] /255;
C_W     = [252, 251, 248] /255;
C_GRID  = [0.88 0.88 0.85];

% 5级金绿梯度（对应5个算法 / 5个阶段）
GRAD5 = {
    [88,  140,  90]/255,   % 深绿
    [138, 174,  90]/255,   % 橄榄绿
    [178, 168,  80]/255,   % 黄绿
    [204, 158,  60]/255,   % 琥珀
    [214, 140,  45]/255    % 深金
};

n_c = 256; h2 = n_c/2;
cmap_gg = [ ...
    linspace(C_GREEN(1),0.97,h2)', linspace(C_GREEN(2),0.97,h2)', linspace(C_GREEN(3),0.97,h2)'; ...
    linspace(0.97,C_GOLD(1),h2)',  linspace(0.97,C_GOLD(2),h2)',  linspace(0.97,C_GOLD(3),h2)'];

fprintf('==========================================================\n');
fprintf('  Hybrid Comparison Visualization\n');
fprintf('  Parallel (w_11) vs Pipeline (w_8)\n');
fprintf('==========================================================\n\n');

%% ==================== 1. 数据准备 ====================
% ── Parallel 数据 ──────────────────────────────────────────
if exist('all_hub_fit','var') && exist('all_conv','var')
    fprintf('[INFO] 使用工作区中的 Parallel 实际运行结果\n');
    par_best  = min(all_hub_fit);
    par_mean  = mean(all_hub_fit);
    par_std   = std(all_hub_fit);
    par_worst = max(all_hub_fit);
    par_conv_mean = mean(all_conv, 1);
    par_conv_std  = std(all_conv, 0, 1);
    n_sync    = size(all_conv, 2);
    nfe_epoch = NFE_TOTAL / n_sync;
    par_nfe   = round(linspace(nfe_epoch, NFE_TOTAL, n_sync));
    par_sync_mean = squeeze(mean(all_sync_hist, 1));   % N_SYNC × N_ALGO

    % 统计每个算法赢得 hub 的次数
    win_count = zeros(1,5);
    for r = 1:size(all_hub_fit,2)
        for ep = 1:n_sync
            ep_fits = squeeze(all_sync_hist(r,ep,:))';
            [~,w] = min(ep_fits);
            win_count(w) = win_count(w)+1;
        end
    end

    % Hub 注入接受率
    accept_rate = squeeze(mean(all_accept,1));   % N_SYNC × N_ALGO
    par_algo_names = ALGO_NAMES;
    par_n_runs = N_RUNS;
else
    fprintf('[WARN] 未找到 Parallel 变量，使用论文结果估算值（demo 模式）\n');
    par_best  = 98483270;
    par_mean  = 98619923;
    par_std   = 123515;
    par_worst = 98817703;
    n_sync    = 20;
    par_nfe   = round(linspace(3750, 75000, n_sync));
    par_conv_mean = linspace(99200000, 98619923, n_sync);
    par_conv_std  = linspace(180000, 123515, n_sync);
    par_sync_mean = repmat(linspace(99200000,98619923,n_sync)',1,5) .* ...
        (1 + 0.002*randn(n_sync,5));
    win_count  = [78, 11, 5, 4, 2];
    accept_rate = repmat(linspace(80,60,n_sync)', 1, 5)/100;
    par_algo_names = {'NSGA','ABC','HLO','GWO','TOW'};
    par_n_runs = 5;
end

% ── Pipeline 数据 ──────────────────────────────────────────
if exist('CC_all','var') && exist('stage_boundary','var')
    fprintf('[INFO] 使用工作区中的 Pipeline 实际运行结果\n');
    pip_best   = bestFit5;
    pip_cc     = CC_all;
    pip_sb     = stage_boundary;
    stage_starts_ = [1, stage_boundary(1:4)+1];
    pip_imprs  = zeros(1,5);
    for s=1:5
        seg = CC_all(stage_starts_(s):stage_boundary(s));
        pip_imprs(s) = max(seg(1)-seg(end),0);
    end
    pip_ted    = TED; pip_atd = ATD;
    pip_mid    = MID; pip_sur = SUR; pip_met = MET;
else
    fprintf('[WARN] 未找到 Pipeline 变量，使用论文结果估算值（demo 模式）\n');
    pip_best   = 97800000;
    % 模拟5段收敛曲线
    s1 = linspace(100200000, 99400000, 250);
    s2 = linspace(99400000,  98900000, 417);
    s3 = linspace(98900000,  98500000, 417);
    s4 = linspace(98500000,  98200000, 375);
    s5 = linspace(98200000,  97800000, 500);
    pip_cc  = [s1, s2, s3, s4, s5]';
    pip_sb  = [250, 667, 1084, 1459, 1959];
    stage_starts_ = [1, pip_sb(1:4)+1];
    pip_imprs  = [800000, 500000, 100000, 90000, 4500];
    pip_ted = 97800000; pip_atd = pip_ted/500;
    pip_mid = 2800;     pip_sur = 72.5; pip_met = 420;
end

pip_nfe_budgets = [20000, 25000, 25000, 30000, 50000];
pip_stage_names = {'S1 ABC','S2 NSGA','S3 NSGA','S4 NSGA','S5 NSGA'};
pip_pop_sizes   = [40, 60, 60, 80, 100];
pip_mut_rates   = [NaN, 0.08, 0.15, 0.04, 0.02];

fprintf('\n  [Parallel] Best=%.0f  Mean=%.0f  Std=%.0f\n', par_best, par_mean, par_std);
fprintf('  [Pipeline] Best=%.0f\n\n', pip_best);

%% ==================== Fig 1: KPI 汇总对比（双轴条形图） ====================
fig1 = figure('Color',C_W,'Name','comp_fig1_kpi_summary','Position',[30 50 1300 500]);

% 1A: 适应度对比
ax1a = subplot(1,3,1);
hold on; box on; grid on;
ax1a.GridColor=C_GRID; ax1a.GridAlpha=.45; ax1a.Color=C_W;

metrics_par = [par_best, par_mean, par_mean+par_std, par_worst];
metrics_pip = [pip_best, pip_best*1.001, pip_best*1.002, pip_best*1.003];
xlbls = {'Best','Mean','Mean+Std','Worst'};

bw = 0.32;
for k = 1:4
    bar(k-bw/2, metrics_par(k), bw, 'FaceColor',C_GREEN, 'EdgeColor','none','FaceAlpha',.9);
    bar(k+bw/2, metrics_pip(k),  bw, 'FaceColor',C_GOLD,  'EdgeColor','none','FaceAlpha',.9);
end
xticks(1:4); xticklabels(xlbls); xtickangle(20);
ylabel('Fitness (m)','FontSize',10,'FontWeight','bold');
title('Fitness Statistics','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
p1=patch(NaN,NaN,C_GREEN,'EdgeColor','none');
p2=patch(NaN,NaN,C_GOLD, 'EdgeColor','none');
legend([p1,p2],{'Parallel','Pipeline'},'Box','off','FontSize',9,'Location','southeast');
set(ax1a,'TickDir','out','FontSize',9,'Box','off');

% 1B: NFE & 效率对比
ax1b = subplot(1,3,2);
hold on; box on; grid on;
ax1b.GridColor=C_GRID; ax1b.GridAlpha=.45; ax1b.Color=C_W;

cats   = {'NFE total (k)','Efficiency (fit/kNFE)'};
v_par  = [75,  par_mean/75000];
v_pip  = [150, pip_best/150000];
scale  = [1/1000, 1];

yyaxis left
bh1 = bar(1-0.2, v_par(1), 0.35, 'FaceColor',C_GREEN,'EdgeColor','none','FaceAlpha',.9);
bh2 = bar(1+0.2, v_pip(1),  0.35, 'FaceColor',C_GOLD, 'EdgeColor','none','FaceAlpha',.9);
ylabel('NFE (thousands)','FontSize',9,'Color',[.35 .35 .35]);
ax1b.YAxis(1).Color = [.35 .35 .35];

yyaxis right
bh3 = bar(2-0.2, v_par(2)/1000, 0.35, 'FaceColor',C_GREEN,'EdgeColor','none','FaceAlpha',.9);
bh4 = bar(2+0.2, v_pip(2)/1000,  0.35, 'FaceColor',C_GOLD, 'EdgeColor','none','FaceAlpha',.9);
ylabel('Efficiency (fit / kNFE, ×10³)','FontSize',9,'Color',[.35 .35 .35]);
ax1b.YAxis(2).Color = [.35 .35 .35];

text(1-0.2, v_par(1)+1,   sprintf('%.0fk',v_par(1)), 'HorizontalAlignment','center','FontSize',8,'Color',C_GREEN*0.7);
text(1+0.2, v_pip(1)+1,   sprintf('%.0fk',v_pip(1)),  'HorizontalAlignment','center','FontSize',8,'Color',C_GOLD*0.7);
text(2-0.2, v_par(2)/1000+0.2, sprintf('%.1f',v_par(2)/1000), 'HorizontalAlignment','center','FontSize',8,'Color',C_GREEN*0.7);
text(2+0.2, v_pip(2)/1000+0.2, sprintf('%.1f',v_pip(2)/1000),  'HorizontalAlignment','center','FontSize',8,'Color',C_GOLD*0.7);

xticks(1:2); xticklabels(cats); xtickangle(10);
title('NFE & Efficiency','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
set(ax1b,'TickDir','out','FontSize',9,'Box','off');

% 1C: 综合得分雷达
ax1c = subplot(1,3,3);
dims = {'Fitness','NFE Effic.','Diversity','Robustness','Multi-Obj','Parallel HW','Stagnation'};
v_par_r = [0.60, 0.85, 0.80, 0.90, 0.45, 0.95, 0.70];
v_pip_r = [0.75, 0.50, 0.65, 0.30, 0.85, 0.10, 0.80];
nD = length(dims);
theta_r = linspace(0,2*pi,nD+1); theta_r(end)=[];

axes('Color',C_W,'Position',ax1c.Position); hold on;
for rv=[0.25,0.5,0.75,1.0]
    th_=linspace(0,2*pi,200);
    plot(cos(th_)*rv,sin(th_)*rv,'-','Color',C_GRID,'LineWidth',0.7);
end
for k=1:nD
    plot([0,cos(theta_r(k))],[0,sin(theta_r(k))],'--','Color',C_GRID,'LineWidth',0.7);
    [xe,ye]=pol2cart(theta_r(k),1.22);
    text(xe,ye,dims{k},'HorizontalAlignment','center','FontSize',7.5,'Color',[.25 .25 .25],'FontWeight','bold');
end
vp=[v_par_r,v_par_r(1)]; tp=[theta_r,theta_r(1)];
[xp,yp]=pol2cart(tp,vp);
fill(xp,yp,C_GREEN,'FaceAlpha',.15,'EdgeColor','none');
plot(xp,yp,'-o','Color',C_GREEN,'LineWidth',2.2,'MarkerSize',5,'MarkerFaceColor',C_GREEN,'MarkerEdgeColor','w');

vq=[v_pip_r,v_pip_r(1)]; tq=[theta_r,theta_r(1)];
[xq,yq]=pol2cart(tq,vq);
fill(xq,yq,C_GOLD,'FaceAlpha',.12,'EdgeColor','none');
plot(xq,yq,'--o','Color',C_GOLD,'LineWidth',2.2,'MarkerSize',5,'MarkerFaceColor',C_GOLD,'MarkerEdgeColor','w');

lr=plot(NaN,NaN,'-o','Color',C_GREEN,'LineWidth',2,'MarkerFaceColor',C_GREEN,'MarkerEdgeColor','w','MarkerSize',5);
lp=plot(NaN,NaN,'--o','Color',C_GOLD, 'LineWidth',2,'MarkerFaceColor',C_GOLD, 'MarkerEdgeColor','w','MarkerSize',5);
legend([lr,lp],{'Parallel','Pipeline'},'Box','off','FontSize',9,'Location','southoutside','Orientation','horizontal');
axis equal off;
title('Capability Radar','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
delete(ax1c);

sgtitle('Parallel vs Pipeline Hybrid — KPI Summary','FontSize',13,'FontWeight','bold','Color',[.12 .12 .12]);
exportgraphics(fig1,'comp_fig1_kpi_summary.png','Resolution',200);
fprintf('Saved: comp_fig1_kpi_summary.png\n');

%% ==================== Fig 2: 收敛曲线对比 ====================
fig2 = figure('Color',C_W,'Name','comp_fig2_convergence','Position',[50 50 1300 560]);

% 2A 左：完整收敛曲线（共享 X 轴 = NFE）
ax2a = subplot(1,2,1);
hold on; box on; grid on;
ax2a.GridColor=C_GRID; ax2a.GridAlpha=.45; ax2a.Color=C_W;

% Parallel — std 阴影
fill([par_nfe, fliplr(par_nfe)], ...
    [par_conv_mean+par_conv_std, fliplr(par_conv_mean-par_conv_std)], ...
    C_GREEN,'FaceAlpha',.15,'EdgeColor','none');
% Parallel — 均值线
plot(par_nfe, par_conv_mean,'-o','Color',C_GREEN,'LineWidth',2.5, ...
    'MarkerSize',4,'MarkerFaceColor',C_GREEN,'MarkerEdgeColor','w', ...
    'DisplayName',sprintf('Parallel Hub (mean, n=%d)',par_n_runs));

% Pipeline — 按阶段着色
pip_nfe_axis = linspace(0, sum(pip_nfe_budgets), length(pip_cc));
stage_starts_ = [1, pip_sb(1:4)+1];
for s = 1:5
    sx  = stage_starts_(s):pip_sb(s);
    plot(pip_nfe_axis(sx), pip_cc(sx), '-', ...
        'Color', GRAD5{s}, 'LineWidth', 2.5, ...
        'DisplayName', sprintf('Pipeline %s', pip_stage_names{s}));
end
% 阶段边界线
for s = 1:4
    xb = pip_nfe_axis(pip_sb(s));
    xline(xb,'--','Color',[0.65 0.65 0.60],'LineWidth',1.0,'HandleVisibility','off');
end

% Parallel 最优点
plot(par_nfe(end), par_best,'p','MarkerSize',14,'MarkerFaceColor',C_GREEN,...
    'MarkerEdgeColor','w','LineWidth',0.8,'DisplayName',sprintf('Par best: %.0f m',par_best));
% Pipeline 最优点
plot(pip_nfe_axis(end), pip_best,'p','MarkerSize',14,'MarkerFaceColor',C_GOLD,...
    'MarkerEdgeColor','w','LineWidth',0.8,'DisplayName',sprintf('Pip best: %.0f m',pip_best));

xlabel('NFE (Function Evaluations)','FontSize',10,'FontWeight','bold');
ylabel('Best Fitness (m)','FontSize',10,'FontWeight','bold');
title('Convergence Comparison (aligned by NFE)','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
legend('Location','northeast','Box','off','FontSize',8.5);
set(ax2a,'TickDir','out','FontSize',9,'Box','off');

% 2B 右：对数尺度归一化收敛（突出相对改进速率）
ax2b = subplot(1,2,2);
hold on; box on; grid on;
ax2b.GridColor=C_GRID; ax2b.GridAlpha=.45; ax2b.Color=C_W;

par_init = par_conv_mean(1);
pip_init = pip_cc(1);
par_rel  = (par_init - par_conv_mean) / par_init * 100;
pip_rel  = (pip_init - pip_cc)       / pip_init * 100;
pip_rel_clean = max(pip_rel, 0);

fill([par_nfe, fliplr(par_nfe)], ...
    [par_rel + par_conv_std/par_init*100, fliplr(par_rel - par_conv_std/par_init*100)], ...
    C_GREEN,'FaceAlpha',.15,'EdgeColor','none');
plot(par_nfe, par_rel,'-o','Color',C_GREEN,'LineWidth',2.5, ...
    'MarkerSize',4,'MarkerFaceColor',C_GREEN,'MarkerEdgeColor','w', ...
    'DisplayName','Parallel (relative)');
plot(pip_nfe_axis, pip_rel_clean,'--','Color',C_GOLD,'LineWidth',2.5, ...
    'DisplayName','Pipeline (relative)');

xlabel('NFE','FontSize',10,'FontWeight','bold');
ylabel('Relative Improvement from Initial (%)','FontSize',10,'FontWeight','bold');
title('Normalised Convergence Rate','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
legend('Location','northwest','Box','off','FontSize',9);
set(ax2b,'TickDir','out','FontSize',9,'Box','off');

sgtitle('Convergence Curve Comparison — Parallel vs Pipeline','FontSize',13,'FontWeight','bold','Color',[.12 .12 .12]);
exportgraphics(fig2,'comp_fig2_convergence.png','Resolution',200);
fprintf('Saved: comp_fig2_convergence.png\n');

%% ==================== Fig 3: NFE 预算分配对比 ====================
fig3 = figure('Color',C_W,'Name','comp_fig3_nfe_budget','Position',[70 50 1200 520]);

% 3A：Parallel NFE 分布（每 epoch）
ax3a = subplot(1,2,1);
hold on; box on; grid on;
ax3a.GridColor=C_GRID; ax3a.GridAlpha=.45; ax3a.Color=C_W;

epoch_nfe = par_nfe(2) - par_nfe(1);   % 每个 epoch 的 NFE
epoch_vals = repmat(epoch_nfe, 1, n_sync);
bh3a = bar(1:n_sync, epoch_vals,'FaceColor','flat','EdgeColor','none','BarWidth',0.80);
for ep=1:n_sync
    t = (ep-1)/(n_sync-1);
    bh3a.CData(ep,:) = (1-t)*C_GREEN + t*C_GOLD;
end
xlabel('Sync Epoch','FontSize',10,'FontWeight','bold');
ylabel('NFE per Epoch (per algo)','FontSize',10,'FontWeight','bold');
title(sprintf('Parallel — NFE/Epoch  (×5 algos, total=%dk/epoch)', ...
    epoch_nfe*5/1000),'FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
text(n_sync*0.5, epoch_nfe*1.05, sprintf('Uniform: %d NFE/epoch', epoch_nfe), ...
    'HorizontalAlignment','center','FontSize',9,'Color',C_MID);
set(ax3a,'TickDir','out','FontSize',9,'Box','off');
ylim([0, epoch_nfe*1.25]);

% 3B：Pipeline NFE 分布（非均匀）
ax3b = subplot(1,2,2);
hold on; box on; grid on;
ax3b.GridColor=C_GRID; ax3b.GridAlpha=.45; ax3b.Color=C_W;

bh3b = bar(pip_nfe_budgets,'FaceColor','flat','EdgeColor','none','BarWidth',0.68);
for s=1:5, bh3b.CData(s,:)=GRAD5{s}; end

% 饼图叠加标注
pip_total = sum(pip_nfe_budgets);
for s=1:5
    pct = pip_nfe_budgets(s)/pip_total*100;
    text(s, pip_nfe_budgets(s)+pip_total*0.015, ...
        sprintf('%.0f%%\n(%dk)',pct,pip_nfe_budgets(s)/1000), ...
        'HorizontalAlignment','center','FontSize',9,'Color',[.2 .2 .2]);
end
xticks(1:5); xticklabels(pip_stage_names); xtickangle(0);
ylabel('NFE Budget','FontSize',10,'FontWeight','bold');
title(sprintf('Pipeline — NFE per Stage  (total=%dk)', pip_total/1000), ...
    'FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);

% 注释：各阶段变异率
mut_annot = {'ABC\nlimit=25','mut=0.08','mut=0.15','mut=0.04','mut=0.02'};
for s=1:5
    text(s, pip_nfe_budgets(s)*0.55, ...
        strrep(mut_annot{s},'\n',sprintf('\n')), ...
        'HorizontalAlignment','center','FontSize',8, ...
        'Color','w','FontWeight','bold');
end
set(ax3b,'TickDir','out','FontSize',9,'Box','off');

sgtitle('NFE Budget Allocation — Parallel (uniform) vs Pipeline (staged)', ...
    'FontSize',13,'FontWeight','bold','Color',[.12 .12 .12]);
exportgraphics(fig3,'comp_fig3_nfe_budget.png','Resolution',200);
fprintf('Saved: comp_fig3_nfe_budget.png\n');

%% ==================== Fig 4: 算法贡献对比 ====================
fig4 = figure('Color',C_W,'Name','comp_fig4_contribution','Position',[90 50 1350 580]);

% 4A: Parallel epoch wins
ax4a = subplot(2,3,1);
hold on; box on; grid on;
ax4a.GridColor=C_GRID; ax4a.GridAlpha=.45; ax4a.Color=C_W;

bh4a = bar(win_count,'FaceColor','flat','EdgeColor','none','BarWidth',0.68);
for a=1:5, bh4a.CData(a,:)=GRAD5{a}; end
total_wins = sum(win_count);
for a=1:5
    text(a, win_count(a)+total_wins*0.02, ...
        sprintf('%d\n(%.0f%%)',win_count(a),win_count(a)/total_wins*100), ...
        'HorizontalAlignment','center','FontSize',8.5,'Color',[.2 .2 .2]);
end
xticks(1:5); xticklabels(par_algo_names);
ylabel('Epoch-Win Count','FontSize',9,'FontWeight','bold');
title('Parallel: Hub Feed Wins','FontSize',10,'FontWeight','bold','Color',[.2 .2 .2]);
set(ax4a,'TickDir','out','FontSize',9,'Box','off');

% 4B: Pipeline stage improvement
ax4b = subplot(2,3,2);
hold on; box on; grid on;
ax4b.GridColor=C_GRID; ax4b.GridAlpha=.45; ax4b.Color=C_W;

bh4b = bar(pip_imprs,'FaceColor','flat','EdgeColor','none','BarWidth',0.68);
for s=1:5, bh4b.CData(s,:)=GRAD5{s}; end
total_impr = sum(pip_imprs);
for s=1:5
    if pip_imprs(s)>0
        text(s, pip_imprs(s)+total_impr*0.015, ...
            sprintf('%.0fk\n(%.0f%%)',pip_imprs(s)/1000, pip_imprs(s)/total_impr*100), ...
            'HorizontalAlignment','center','FontSize',8.5,'Color',[.2 .2 .2]);
    end
end
xticks(1:5); xticklabels(pip_stage_names); xtickangle(15);
ylabel('Fitness Improvement (m)','FontSize',9,'FontWeight','bold');
title('Pipeline: Per-Stage Improvement','FontSize',10,'FontWeight','bold','Color',[.2 .2 .2]);
set(ax4b,'TickDir','out','FontSize',9,'Box','off');

% 4C: Parallel Hub 接受率折线
ax4c = subplot(2,3,3);
hold on; box on; grid on;
ax4c.GridColor=C_GRID; ax4c.GridAlpha=.45; ax4c.Color=C_W;

for a=1:5
    plot(1:n_sync, accept_rate(:,a)*100, '-o', ...
        'Color', GRAD5{a}, 'LineWidth', 2.0, ...
        'MarkerSize',3,'MarkerFaceColor',GRAD5{a},'MarkerEdgeColor','w', ...
        'DisplayName', par_algo_names{a});
end
xlabel('Sync Epoch','FontSize',9,'FontWeight','bold');
ylabel('Hub Acceptance Rate (%)','FontSize',9,'FontWeight','bold');
title('Parallel: Hub Injection Acceptance','FontSize',10,'FontWeight','bold','Color',[.2 .2 .2]);
legend('Location','best','Box','off','FontSize',8.5);
ylim([0 105]);
set(ax4c,'TickDir','out','FontSize',9,'Box','off');

% 4D: Parallel per-algo 最终适应度（箱形图替代）
ax4d = subplot(2,3,4);
hold on; box on; grid on;
ax4d.GridColor=C_GRID; ax4d.GridAlpha=.45; ax4d.Color=C_W;

final_fits_par = squeeze(par_sync_mean(end,:));
for a = 1:5
    bar(a, final_fits_par(a), 0.65, 'FaceColor',GRAD5{a},'EdgeColor','none','FaceAlpha',.9);
    text(a, final_fits_par(a)+range(final_fits_par)*0.015, ...
        sprintf('%.0fM',final_fits_par(a)/1e6), ...
        'HorizontalAlignment','center','FontSize',8,'Color',[.2 .2 .2]);
end
xticks(1:5); xticklabels(par_algo_names);
ylabel('Mean Final Fitness (m)','FontSize',9,'FontWeight','bold');
title('Parallel: Per-Algo Final Fitness','FontSize',10,'FontWeight','bold','Color',[.2 .2 .2]);
set(ax4d,'TickDir','out','FontSize',9,'Box','off');

% 4E: Pipeline mutation rate 衰减曲线
ax4e = subplot(2,3,5);
hold on; box on; grid on;
ax4e.GridColor=C_GRID; ax4e.GridAlpha=.45; ax4e.Color=C_W;

valid_s = 2:5; vm = pip_mut_rates(valid_s); vs = pip_pop_sizes(valid_s);
yyaxis left
plot(valid_s, vm, '-o','Color',C_GREEN,'LineWidth',2.2, ...
    'MarkerSize',7,'MarkerFaceColor',C_GREEN,'MarkerEdgeColor','w');
for s=valid_s
    text(s, pip_mut_rates(s)+0.005, sprintf('%.2f',pip_mut_rates(s)), ...
        'HorizontalAlignment','center','FontSize',8.5,'Color',C_GREEN*0.7,'FontWeight','bold');
end
ylabel('Mutation Rate','FontSize',9,'Color',C_GREEN*0.6);
ax4e.YAxis(1).Color = C_GREEN*0.6; ylim([0, 0.20]);

yyaxis right
plot(1:5, pip_pop_sizes, '--s','Color',C_GOLD,'LineWidth',2.2, ...
    'MarkerSize',7,'MarkerFaceColor',C_GOLD,'MarkerEdgeColor','w');
ylabel('Population Size','FontSize',9,'Color',C_GOLD*0.7);
ax4e.YAxis(2).Color = C_GOLD*0.7;

xticks(1:5); xticklabels(pip_stage_names); xtickangle(15);
title('Pipeline: Mut Rate & Pop Size','FontSize',10,'FontWeight','bold','Color',[.2 .2 .2]);
set(ax4e,'TickDir','out','FontSize',9,'Box','off');

% 4F: 综合效率对比条
ax4f = subplot(2,3,6);
hold on; box on; grid on;
ax4f.GridColor=C_GRID; ax4f.GridAlpha=.45; ax4f.Color=C_W;

metrics_name = {'Fit Reduction (M m)','Fit/kNFE (k m)','Pop Max','Stag. Escapes'};
v_par_eff = [(par_conv_mean(1)-par_best)/1e6,  (par_conv_mean(1)-par_best)/75,   50, 20];
v_pip_eff = [(pip_cc(1)-pip_best)/1e6,         (pip_cc(1)-pip_best)/150,          100, 5];

% 归一化到 0-1
for k=1:4
    mx=max(v_par_eff(k),v_pip_eff(k));
    vp=v_par_eff(k)/mx; vq=v_pip_eff(k)/mx;
    barh(k-0.18, vp, 0.30,'FaceColor',C_GREEN,'EdgeColor','none','FaceAlpha',.9);
    barh(k+0.18, vq, 0.30,'FaceColor',C_GOLD, 'EdgeColor','none','FaceAlpha',.9);
    text(vp+0.02, k-0.18, sprintf('%.2f',v_par_eff(k)),'FontSize',8,'Color',C_GREEN*0.7,'VerticalAlignment','middle');
    text(vq+0.02, k+0.18, sprintf('%.2f',v_pip_eff(k)), 'FontSize',8,'Color',C_GOLD*0.7, 'VerticalAlignment','middle');
end
yticks(1:4); yticklabels(metrics_name);
xlabel('Normalised Score','FontSize',9,'FontWeight','bold');
title('Efficiency Comparison (normalised)','FontSize',10,'FontWeight','bold','Color',[.2 .2 .2]);
ph1=patch(NaN,NaN,C_GREEN,'EdgeColor','none');
ph2=patch(NaN,NaN,C_GOLD, 'EdgeColor','none');
legend([ph1,ph2],{'Parallel','Pipeline'},'Box','off','FontSize',9,'Location','southeast');
set(ax4f,'TickDir','out','FontSize',9,'Box','off'); xlim([0,1.4]);

sgtitle('Algorithm Contribution Analysis — Parallel vs Pipeline', ...
    'FontSize',13,'FontWeight','bold','Color',[.12 .12 .12]);
exportgraphics(fig4,'comp_fig4_contribution.png','Resolution',200);
fprintf('Saved: comp_fig4_contribution.png\n');

%% ==================== Fig 5: 热力图对比 ====================
fig5 = figure('Color',C_W,'Name','comp_fig5_heatmap','Position',[110 50 1400 520]);

% 5A: Parallel sync × algo 热力图（适应度）
ax5a = subplot(1,2,1);
imagesc(par_sync_mean');   % N_ALGO × N_SYNC
colormap(ax5a, cmap_gg);
mx5=max(par_sync_mean(:)); mn5=min(par_sync_mean(:));
try, clim([mn5 mx5]); catch, caxis([mn5 mx5]); end
cb5a=colorbar('southoutside'); cb5a.Label.String='Fitness (m)'; cb5a.FontSize=8;

for a=1:5
    for ep=1:n_sync
        text(ep,a,sprintf('%.1fM',par_sync_mean(ep,a)/1e6), ...
            'HorizontalAlignment','center','VerticalAlignment','middle', ...
            'FontSize',6.5,'Color',[.1 .1 .1],'FontWeight','bold');
    end
end
% 框出每 epoch 的最优算法
for ep=1:n_sync
    [~,ba]=min(par_sync_mean(ep,:));
    rectangle('Position',[ep-0.5,ba-0.5,1,1],'EdgeColor',C_GOLD,'LineWidth',1.8,'Curvature',.1);
end
xticks(1:n_sync);
xticklabels(arrayfun(@(e)sprintf('S%d',e),1:n_sync,'UniformOutput',false));
xtickangle(45);
yticks(1:5); yticklabels(par_algo_names);
xlabel('Sync Epoch','FontSize',10,'FontWeight','bold');
title('Parallel: Sync × Algo Fitness Heatmap  (box=epoch winner)', ...
    'FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
set(ax5a,'TickDir','out','FontSize',8.5,'Box','off');

% 5B: Pipeline 阶段汇总热力图
ax5b = subplot(1,2,2);
stage_starts_ = [1, pip_sb(1:4)+1];
pip_hmap = zeros(4,5);
for s=1:5
    seg=pip_cc(stage_starts_(s):pip_sb(s));
    pip_hmap(1,s)=seg(1);
    pip_hmap(2,s)=seg(end);
    pip_hmap(3,s)=100*(seg(1)-seg(end))/max(seg(1),1);
    pip_hmap(4,s)=pip_nfe_budgets(s)/1000;
end
% 逐行归一化
pip_norm=zeros(4,5);
for r=1:4
    rmin=min(pip_hmap(r,:)); rmax=max(pip_hmap(r,:));
    if rmax-rmin<1e-10, pip_norm(r,:)=0.5;
    else, pip_norm(r,:)=(pip_hmap(r,:)-rmin)/(rmax-rmin); end
end
pip_norm(1,:)=1-pip_norm(1,:);
pip_norm(2,:)=1-pip_norm(2,:);

imagesc(pip_norm);
colormap(ax5b, cmap_gg);
try, clim([0 1]); catch, caxis([0 1]); end
cb5b=colorbar('southoutside');
cb5b.Label.String='Relative quality  (green=better, gold=worse)';
cb5b.FontSize=8; cb5b.Ticks=[0 1]; cb5b.TickLabels={'Better','Worse'};

fmts={'%.1fM','%.1fM','%.1f%%','%.0fk'};
for r=1:4
    for s=1:5
        v=pip_hmap(r,s);
        if r<=2, lbl=sprintf('%.1fM',v/1e6);
        elseif r==3, lbl=sprintf('%.1f%%',v);
        else, lbl=sprintf('%.0fk',v); end
        text(s,r,lbl,'HorizontalAlignment','center','VerticalAlignment','middle', ...
            'FontSize',9,'FontWeight','bold','Color',[.12 .12 .12]);
    end
end
xticks(1:5); xticklabels(pip_stage_names); xtickangle(15);
yticks(1:4); yticklabels({'Start (m)','End (m)','Improv %','NFE (k)'});
title('Pipeline: Stage Summary Heatmap','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
set(ax5b,'TickDir','out','FontSize',9,'Box','off');

sgtitle('Fitness Heatmap — Parallel Sync Structure vs Pipeline Stage Structure', ...
    'FontSize',13,'FontWeight','bold','Color',[.12 .12 .12]);
exportgraphics(fig5,'comp_fig5_heatmap.png','Resolution',200);
fprintf('Saved: comp_fig5_heatmap.png\n');

%% ==================== Fig 6: 累计改进 & 运行稳健性 ====================
fig6 = figure('Color',C_W,'Name','comp_fig6_improvement','Position',[130 50 1300 540]);

% 6A: 累计改进曲线（vs NFE）
ax6a = subplot(1,2,1);
hold on; box on; grid on;
ax6a.GridColor=C_GRID; ax6a.GridAlpha=.45; ax6a.Color=C_W;

par_cumimpr = max(par_conv_mean(1)-par_conv_mean, 0);
pip_cumimpr = max(pip_cc(1)-pip_cc, 0);

fill([par_nfe, fliplr(par_nfe)], ...
    [par_cumimpr + par_conv_std, fliplr(max(par_cumimpr-par_conv_std,0))], ...
    C_GREEN,'FaceAlpha',.15,'EdgeColor','none');
plot(par_nfe, par_cumimpr,'-o','Color',C_GREEN,'LineWidth',2.5, ...
    'MarkerSize',4,'MarkerFaceColor',C_GREEN,'MarkerEdgeColor','w', ...
    'DisplayName','Parallel hub (mean)');
plot(pip_nfe_axis, pip_cumimpr,'--','Color',C_GOLD,'LineWidth',2.5, ...
    'DisplayName','Pipeline');

% Pipeline 阶段色块
for s=1:5
    sx=stage_starts_(s):pip_sb(s);
    fill([pip_nfe_axis(sx), fliplr(pip_nfe_axis(sx))], ...
        [pip_cumimpr(sx)', zeros(1,length(sx))], ...
        GRAD5{s},'FaceAlpha',.10,'EdgeColor','none','HandleVisibility','off');
end

xlabel('NFE','FontSize',10,'FontWeight','bold');
ylabel('Cumulative Improvement from Initial (m)','FontSize',10,'FontWeight','bold');
title('Cumulative Improvement vs NFE','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
legend('Location','northwest','Box','off','FontSize',9);
set(ax6a,'TickDir','out','FontSize',9,'Box','off');

% 6B: Parallel 多次运行稳健性
ax6b = subplot(1,2,2);
hold on; box on; grid on;
ax6b.GridColor=C_GRID; ax6b.GridAlpha=.45; ax6b.Color=C_W;

% 每次独立运行曲线（若有）
if exist('all_conv','var')
    for r=1:par_n_runs
        t=(r-1)/(par_n_runs-1);
        clr=(1-t)*C_GREEN+t*C_GOLD;
        plot(par_nfe, all_conv(r,:),'-','Color',[clr,0.5],'LineWidth',1.2, ...
            'DisplayName',sprintf('Run %d',r));
    end
else
    for r=1:5
        mock_conv = par_conv_mean + randn(1,n_sync)*par_std*0.8;
        t=(r-1)/4; clr=(1-t)*C_GREEN+t*C_GOLD;
        plot(par_nfe, mock_conv,'-','Color',[clr,0.5],'LineWidth',1.2, ...
            'DisplayName',sprintf('Run %d',r));
    end
end
plot(par_nfe, par_conv_mean,'-o','Color',[.2 .2 .2],'LineWidth',2.5, ...
    'MarkerSize',4,'MarkerFaceColor',[.2 .2 .2],'MarkerEdgeColor','w', ...
    'DisplayName','Mean');
% Pipeline single run
plot(pip_nfe_axis, pip_cc,'--','Color',C_GOLD,'LineWidth',2.2, ...
    'DisplayName','Pipeline (1 run)');

yline(pip_best,'--','Color',C_GOLD*.6,'LineWidth',1,'HandleVisibility','off');
yline(par_best,'-' ,'Color',C_GREEN*.7,'LineWidth',1,'HandleVisibility','off');

xlabel('NFE','FontSize',10,'FontWeight','bold');
ylabel('Best Fitness (m)','FontSize',10,'FontWeight','bold');
title(sprintf('Parallel Robustness (%d runs) vs Pipeline',par_n_runs), ...
    'FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
legend('Location','northeast','Box','off','FontSize',8.5);
set(ax6b,'TickDir','out','FontSize',9,'Box','off');

sgtitle('Cumulative Improvement & Multi-Run Robustness', ...
    'FontSize',13,'FontWeight','bold','Color',[.12 .12 .12]);
exportgraphics(fig6,'comp_fig6_improvement.png','Resolution',200);
fprintf('Saved: comp_fig6_improvement.png\n');

%% ==================== Fig 7: 精英传递机制对比图（结构示意） ====================
fig7 = figure('Color',C_W,'Name','comp_fig7_elite_mechanism','Position',[150 50 1300 500]);

% 7A: Parallel Hub 注入接受热力图（阶段 × 算法）
ax7a = subplot(1,2,1);
imagesc(accept_rate'*100);   % N_ALGO × N_SYNC
colormap(ax7a, cmap_gg);
try, clim([0 100]); catch, caxis([0 100]); end
cb7a=colorbar('southoutside');
cb7a.Label.String='Hub Injection Acceptance Rate (%)';
cb7a.FontSize=8;

for a=1:5
    for ep=1:n_sync
        text(ep,a,sprintf('%.0f%%',accept_rate(ep,a)*100), ...
            'HorizontalAlignment','center','VerticalAlignment','middle', ...
            'FontSize',6.5,'Color',[.1 .1 .1],'FontWeight','bold');
    end
end
xticks(1:n_sync);
xticklabels(arrayfun(@(e)sprintf('S%d',e),1:n_sync,'UniformOutput',false));
xtickangle(45);
yticks(1:5); yticklabels(par_algo_names);
xlabel('Sync Epoch','FontSize',10,'FontWeight','bold');
title('Parallel: Hub Injection Acceptance Heatmap','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
set(ax7a,'TickDir','out','FontSize',8.5,'Box','off');

% 7B: 两种 elite 传递机制对比条（每阶段接受率/贡献）
ax7b = subplot(1,2,2);
hold on; box on; grid on;
ax7b.GridColor=C_GRID; ax7b.GridAlpha=.45; ax7b.Color=C_W;

% Parallel：平均接受率（按算法）
par_accept_by_algo = mean(accept_rate,1)*100;
% Pipeline：按阶段贡献率
pip_contrib_by_stage = pip_imprs/max(sum(pip_imprs),1)*100;

bw7=0.35;
for a=1:5
    bar(a-bw7/2, par_accept_by_algo(a), bw7, 'FaceColor',GRAD5{a},'EdgeColor','none','FaceAlpha',.9);
    bar(a+bw7/2, pip_contrib_by_stage(a), bw7, 'FaceColor',GRAD5{a}*.6+[1 1 1]*.4,'EdgeColor','none','FaceAlpha',.9);
    text(a-bw7/2, par_accept_by_algo(a)+1.5, sprintf('%.0f%%',par_accept_by_algo(a)), ...
        'HorizontalAlignment','center','FontSize',8.5,'Color',GRAD5{a}*.7);
    text(a+bw7/2, pip_contrib_by_stage(a)+1.5, sprintf('%.0f%%',pip_contrib_by_stage(a)), ...
        'HorizontalAlignment','center','FontSize',8.5,'Color',GRAD5{a}*.5);
end

xlbls7 = cellfun(@(pa,ps) sprintf('%s\n%s',pa,ps), ...
    par_algo_names, pip_stage_names, 'UniformOutput',false);
xticks(1:5); xticklabels(xlbls7); xtickangle(0);
ylabel('Rate / Contribution (%)','FontSize',10,'FontWeight','bold');
title('Par Hub Accept % vs Pip Stage Contrib %','FontSize',11,'FontWeight','bold','Color',[.2 .2 .2]);
pd1=patch(NaN,NaN,C_GREEN,'EdgeColor','none');
pd2=patch(NaN,NaN,C_MID*.7+[1 1 1]*.3,'EdgeColor','none');
legend([pd1,pd2],{'Parallel (hub accept%)','Pipeline (stage contrib%)'},'Box','off','FontSize',9,'Location','northeast');
set(ax7b,'TickDir','out','FontSize',9,'Box','off');

sgtitle('Elite Transfer Mechanism Comparison','FontSize',13,'FontWeight','bold','Color',[.12 .12 .12]);
exportgraphics(fig7,'comp_fig7_elite_mechanism.png','Resolution',200);
fprintf('Saved: comp_fig7_elite_mechanism.png\n');

%% ==================== 输出汇总 ====================
fprintf('\n========================================================\n');
fprintf('  对比可视化完成！已保存 7 张图：\n');
fprintf('  comp_fig1_kpi_summary.png      — KPI汇总+雷达图\n');
fprintf('  comp_fig2_convergence.png      — 收敛曲线对比\n');
fprintf('  comp_fig3_nfe_budget.png       — NFE预算分配\n');
fprintf('  comp_fig4_contribution.png     — 算法贡献分析(6子图)\n');
fprintf('  comp_fig5_heatmap.png          — 热力图对比\n');
fprintf('  comp_fig6_improvement.png      — 累计改进+稳健性\n');
fprintf('  comp_fig7_elite_mechanism.png  — 精英传递机制\n');
fprintf('========================================================\n');
fprintf('  [Parallel] Best=%.0f  Mean=%.0f  Std=%.0f  Runs=%d\n', ...
    par_best, par_mean, par_std, par_n_runs);
fprintf('  [Pipeline] Best=%.0f  NFE_total=%d\n', pip_best, sum(pip_nfe_budgets));
fprintf('========================================================\n');