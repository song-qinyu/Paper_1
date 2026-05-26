%% ============================================================
%  Pipeline Hybrid — Stage 竞选框架 v2（全15算法版）
%  + 完整 SHAP 风格评估可视化（7张图）
%
%  算法列表：GA, DE, PSO, SSA, GWO, FA, ABC, TOW, CA, PO, CS, HLO, SA, HS, NSGA
%
%  可视化输出（10张图）：
%    [原始]  各阶段竞选结果 + Pipeline趋势
%    [SHAP]  fig1_election_ranking.png     各阶段横向条形排名 + 折线趋势
%            fig2_shap_beeswarm.png        SHAP Beeswarm（算法×阶段贡献分布）
%            fig3_main_vs_interaction.png  主效应 vs 交互效应柱图
%            fig4_shap_heatmap.png         SHAP 热力图（算法×阶段）
%            fig5_interaction_matrix.png   算法交互矩阵（15×15）
%            fig6_shap_dependence.png      单算法 Lowess 依赖图
%            fig7_radar.png                多维雷达图（算法综合能力）
%% ============================================================
clc; clear; close all; tic;

fprintf('========================================================\n');
fprintf('  Pipeline Stage 竞选框架 v2 + SHAP 可视化\n');
fprintf('  全15算法公平评测\n');
fprintf('========================================================\n\n');

%% ==================== 0. 数据加载 ====================
if exist('sj5.mat','file'), load('sj5.mat');
else, error('未找到 sj5.mat'); end

if exist('dis','var'), data.dis = dis; end

dim = length(DFenPei);
Lb  = ones(1, dim);
Ub  = arrayfun(@(i) length(DFenPei{i})-1, 1:dim);

FID=[]; alldis_fixed=0;
for k=1:length(B)
    if length(B{k})==1
        FID=[FID;k,B{k}];
        alldis_fixed=alldis_fixed+data.dis(k,B{k});
    end
end

fobj = @(x) unified_fobj(x, DFenPei, data.dis, Lb, Ub);

%% ==================== 1. 预算设置 ====================
NFE_per_algo = 15000;
K_elite      = 20;

%% ==================== 2. 五阶段竞选 ====================
elite_pop = []; elite_fit = [];
all_results = cell(5,1);
stage_labels = {'探索','收缩','脱困','精化','深收敛'};

for stage = 1:5
    fprintf('--- Stage %d: %s阶段 ---\n', stage, stage_labels{stage});
    if stage==1
        fprintf('  （无精英输入，全随机初始化）\n');
    else
        fprintf('  （接收上阶段精英池 %d 个解）\n', size(elite_pop,1));
    end

    results = run_stage_competition(fobj, dim, Lb, Ub, ...
        NFE_per_algo, elite_pop, elite_fit, K_elite);

    all_results{stage} = results;
    fprintf('\n[Stage %d 排名]\n', stage);
    print_ranking(results);
    [elite_pop, elite_fit] = get_elite_pool(results, K_elite, dim);
    fprintf('\n');
end

run_time = toc;

%% ==================== 3. 汇总报告 ====================
fprintf('========================================================\n');
fprintf('  各阶段最优算法汇总\n');
fprintf('========================================================\n');
for s=1:5
    r=all_results{s}; fits=[r.best_fit]; [~,wi]=min(fits);
    fprintf('  Stage %d (%s): 最优算法 = %-6s | 最优值 = %.2f m\n', ...
        s, stage_labels{s}, r(wi).name, r(wi).best_fit);
end
fprintf('\n建议的 Pipeline 组合（数据驱动）：\n');
for s=1:5
    r=all_results{s}; [~,wi]=min([r.best_fit]);
    fprintf('  Stage %d → %s\n', s, r(wi).name);
end
fprintf('\n总评测时间：%.2f 秒\n', run_time);
fprintf('========================================================\n');

%% ==================== 4. 原始竞选结果图 ====================
figure('Color','w','Name','各阶段竞选结果','Position',[50,50,1400,820]);
for s=1:5
    subplot(2,3,s);
    r=all_results{s}; names={r.name}; fits=[r.best_fit];
    [fits_s,si]=sort(fits); names_s=names(si);
    b=barh(fits_s,'FaceColor','flat');
    clrs=repmat([0.75 0.88 1.0],length(fits_s),1);
    clrs(1,:)=[0.18 0.72 0.32]; clrs(2,:)=[0.95 0.85 0.20];
    b.CData=clrs;
    yticks(1:length(names_s)); yticklabels(names_s);
    xlabel('适应度（总距离 m）');
    title(sprintf('Stage %d  %s', s, stage_labels{s}),'FontSize',11,'FontWeight','bold');
    grid on; box on;
    text(fits_s(1)*1.001,1,sprintf(' %.0f ★',fits_s(1)),'FontSize',8,'Color',[0.05 0.45 0.1],'FontWeight','bold');
end
subplot(2,3,6);
stage_bests=zeros(1,5); stage_winners=cell(1,5);
for s=1:5
    r=all_results{s}; [v,wi]=min([r.best_fit]);
    stage_bests(s)=v; stage_winners{s}=r(wi).name;
end
plot(1:5,stage_bests,'-o','LineWidth',2.5,'Color',[0.2 0.5 0.9],...
    'MarkerFaceColor',[0.2 0.5 0.9],'MarkerSize',8);
xticks(1:5);
xticklabels(cellfun(@(s,w) sprintf('S%d\n%s',s,w), num2cell(1:5),stage_winners,'UniformOutput',false));
ylabel('最优适应度 (m)'); title('Pipeline 各阶段最优值趋势','FontSize',11,'FontWeight','bold');
grid on; box on;
sgtitle('Pipeline Hybrid 各阶段算法竞选结果（越短越优）','FontSize',13,'FontWeight','bold');

%% ==================== 5. 构建 SHAP 分析数据矩阵 ====================
% 从竞选结果中提取各算法在各阶段的表现
fprintf('\n构建 SHAP 分析矩阵...\n');

% 收集所有算法名
all_algo_names = {};
for s=1:5
    r=all_results{s};
    for i=1:length(r), all_algo_names{end+1}=r(i).name; end
end
algo_names = unique(all_algo_names,'stable');
nAlgo  = length(algo_names);
nStage = 5;

% 构建性能矩阵 [nAlgo × nStage]（fitness值，越小越好）
perf_mat  = NaN(nAlgo, nStage);
% 归一化改进矩阵（SHAP值，越大越好）
shap_mat  = zeros(nAlgo, nStage);

for s=1:5
    r=all_results{s};
    fits_all=[r.best_fit];
    f_max=max(fits_all); f_min=min(fits_all);
    f_range=max(f_max-f_min, 1);

    for i=1:length(r)
        ai=find(strcmp(algo_names, r(i).name));
        if ~isempty(ai)
            perf_mat(ai,s) = r(i).best_fit;
            % SHAP值 = 归一化相对于最差的改进量（越好分越高）
            shap_mat(ai,s) = (f_max - r(i).best_fit) / f_range;
        end
    end
end

% 全局重要性：每个算法的平均SHAP值
algo_importance = nanmean(shap_mat, 2);
% 按重要性排序（降序）
[~, imp_order] = sort(algo_importance, 'descend');

% 各阶段排名矩阵（1=最优）
rank_mat = zeros(nAlgo, nStage);
for s=1:5
    col=perf_mat(:,s);
    valid=~isnan(col);
    [~,sr]=sort(col(valid));
    idx_v=find(valid);
    for r=1:length(sr), rank_mat(idx_v(sr(r)),s)=r; end
end

% 各阶段胜者标记
stage_winner_idx = zeros(1,5);
for s=1:5
    [~,wi]=min(perf_mat(:,s)); stage_winner_idx(s)=wi;
end

fprintf('  ✓ SHAP矩阵构建完成 [%d算法 × %d阶段]\n\n', nAlgo, nStage);

%% ==================== 6. 颜色系统 ====================
C_green  = [0.20 0.50 0.25];
C_orange = [0.85 0.55 0.10];
C_bg     = [1.00 1.00 1.00];

% 为每个算法分配固定颜色
algo_color_palette = [
    0.12 0.47 0.71;  % GA   — 蓝
    1.00 0.50 0.05;  % DE   — 橙
    0.17 0.63 0.17;  % PSO  — 绿
    0.84 0.15 0.16;  % SSA  — 红
    0.58 0.40 0.74;  % GWO  — 紫
    0.55 0.34 0.29;  % FA   — 棕
    0.89 0.47 0.76;  % ABC  — 粉
    0.50 0.50 0.50;  % TOW  — 灰
    0.74 0.74 0.13;  % CA   — 黄绿
    0.09 0.75 0.81;  % PO   — 青
    0.93 0.23 0.23;  % CS   — 深红
    0.35 0.71 0.35;  % HLO  — 浅绿
    0.20 0.20 0.80;  % SA   — 深蓝
    0.80 0.60 0.10;  % HS   — 金
    0.50 0.10 0.70;  % NSGA — 深紫
];
if size(algo_color_palette,1) < nAlgo
    extra = repmat([0.5 0.5 0.5], nAlgo-size(algo_color_palette,1), 1);
    algo_color_palette = [algo_color_palette; extra];
end

%% ============================================================
%  Fig 1 — 各阶段横向条形排名 + Pipeline折线趋势
%% ============================================================
fprintf('[Fig 1] 各阶段横向条形排名 + 趋势折线...\n');

fig1 = figure('Color',C_bg,'Position',[60,60,1300,560],'Name','Fig1');

for s=1:5
    ax = subplot(2,5,s);
    hold on; box on;
    r=all_results{s};
    fits=[r.best_fit]; names_r={r.name};
    [fits_s,si]=sort(fits,'descend');  % 升序显示（小值在上）
    names_s=names_r(si);
    nA=length(fits_s);
    for i=1:nA
        ai=find(strcmp(algo_names,names_s{i}));
        fc=algo_color_palette(ai,:);
        if i==nA, fc=[0.18 0.72 0.32]; end  % 最优=金绿
        barh(i, fits_s(i), 0.7,'FaceColor',fc,'EdgeColor','none','FaceAlpha',0.85);
    end
    yticks(1:nA); yticklabels(names_s); set(ax,'FontSize',7.5,'TickDir','out');
    xlabel('Fitness (m)','FontSize',8);
    title(sprintf('Stage %d  %s', s, stage_labels{s}),'FontSize',10,'FontWeight','bold');
    grid on;
    text(fits_s(end)*1.001, nA, sprintf('★%.0f',fits_s(end)),...
         'FontSize',7.5,'Color',[0.05 0.45 0.1],'FontWeight','bold','VerticalAlignment','middle');
end

% 下方：各阶段最优值折线 + 胜者标注
ax6 = subplot(2,5,6:10);
hold on; box on; grid on;

% 画每个算法的折线（淡色）
for ai=1:nAlgo
    vals=perf_mat(ai,:);
    valid=~isnan(vals);
    if sum(valid)>1
        plot(find(valid), vals(valid),'.-','Color',[algo_color_palette(ai,:),0.25],...
             'LineWidth',0.8,'MarkerSize',6,'HandleVisibility','off');
    end
end
% 画胜者连线（粗，绿橙交替）
winner_vals=zeros(1,5);
for s=1:5, winner_vals(s)=perf_mat(stage_winner_idx(s),s); end
plot(1:5, winner_vals,'-o','LineWidth',3.0,'Color',[0.15 0.60 0.25],...
     'MarkerFaceColor',[0.15 0.60 0.25],'MarkerEdgeColor','w',...
     'MarkerSize',10,'DisplayName','Stage Winner');

for s=1:5
    wn=algo_names{stage_winner_idx(s)};
    text(s, winner_vals(s)-range(winner_vals)*0.04, sprintf('%s',wn),...
         'HorizontalAlignment','center','FontSize',9,'FontWeight','bold','Color',[0.1 0.4 0.1]);
end

set(ax6,'XTick',1:5,'XTickLabel',stage_labels,'FontSize',10,'GridAlpha',0.2,'TickDir','out');
xlabel('Pipeline Stage','FontWeight','bold','FontSize',11);
ylabel('Best Fitness (m)','FontWeight','bold','FontSize',11);
title('各阶段最优值趋势（胜者 Pipeline）','FontSize',12,'FontWeight','bold');
legend('Location','northeast');

sgtitle('Pipeline Hybrid 竞选结果总览 — 各阶段排名','FontSize',14,'FontWeight','bold');
saveas(fig1,'fig1_election_ranking.png');
fprintf('  ✓ fig1_election_ranking.png\n\n');

%% ============================================================
%  Fig 2 — SHAP Beeswarm（算法 × 阶段贡献分布）
%  纵轴：算法（按全局重要性降序）
%  横轴：SHAP值（相对改进量）
%  点色：该阶段性能（绿=好 橙=差）
%% ============================================================
fprintf('[Fig 2] SHAP Beeswarm...\n');

fig2 = figure('Color',C_bg,'Position',[80,80,900,620],'Name','Fig2');
ax2  = axes; hold on; box on;

% 排序后的算法标签（从下往上 = 不重要到重要）
sorted_names = algo_names(imp_order);
sorted_shap  = shap_mat(imp_order,:);
sorted_perf  = perf_mat(imp_order,:);

rng(42);
stage_marker = {'o','s','^','d','p'};

for ai=1:nAlgo
    for s=1:nStage
        if isnan(sorted_perf(ai,s)), continue; end
        sv   = sorted_shap(ai,s);
        % 用性能值做着色（归一化到 0~1，0=最好=深绿，1=最差=橙）
        col_v= 1 - sv;   % 翻转：SHAP高=性能好=绿
        y_jit= ai + (rand-0.5)*0.40;
        scatter(sv, y_jit, 28, col_v, 'filled',...
                'MarkerFaceAlpha', 0.75, 'MarkerEdgeColor','none',...
                'Marker', stage_marker{s});
    end
    % 均值条
    mv = nanmean(sorted_shap(ai,:));
    plot([mv mv],[ai-0.45,ai+0.45],'-','Color',...
         algo_color_palette(imp_order(ai),:),'LineWidth',2.2);
    text(mv+0.015, ai+0.38, sprintf('%.2f',mv),...
         'FontSize',7.5,'FontWeight','bold',...
         'Color',algo_color_palette(imp_order(ai),:));
end

% 图例（阶段标记）
for s=1:5
    scatter(NaN,NaN,28,'k','filled','Marker',stage_marker{s},...
            'DisplayName',sprintf('Stage%d %s',s,stage_labels{s}));
end

cb2=colorbar('eastoutside');
colormap(ax2, cmap_go(256));
cb2.Label.String='Feature value: Performance (Green=High/Good → Orange=Low/Poor)';
cb2.Label.FontSize=8;
clim([0 1]);
xline(0,'--','Color',[0.6 0.6 0.6],'LineWidth',1.2,'HandleVisibility','off');

set(ax2,'YTick',1:nAlgo,'YTickLabel',sorted_names,'FontSize',9,...
    'XGrid','on','GridAlpha',0.25,'TickDir','out');
xlabel('SHAP Value (Relative Improvement)','FontWeight','bold','FontSize',11);
title({'图 2：算法全局贡献 Beeswarm（SHAP 风格）',...
       '各算法在各阶段的贡献分布（按全局重要性排序）'},...
      'FontSize',12,'FontWeight','bold');
xlim([-0.05, 1.10]); ylim([0.3, nAlgo+0.7]);
legend('Location','southeast','FontSize',8,'Box','on','NumColumns',3);
saveas(fig2,'fig2_shap_beeswarm.png');
fprintf('  ✓ fig2_shap_beeswarm.png\n\n');

%% ============================================================
%  Fig 3 — 主效应 vs 交互效应柱图
%  主效应：算法自身在各阶段的平均SHAP贡献
%  交互效应：算法配合精英传递给下一阶段带来的额外提升（估算）
%% ============================================================
fprintf('[Fig 3] 主效应 vs 交互效应...\n');

fig3 = figure('Color',C_bg,'Position',[100,100,1100,520],'Name','Fig3');
ax3  = axes; hold on; box on; grid on;

main_eff = algo_importance(imp_order);   % 已按重要性排序

% 交互效应：若某算法赢得某阶段，其精英对下阶段的增益
interact_eff = zeros(nAlgo,1);
for ai_s=1:nAlgo
    ai_orig = imp_order(ai_s);
    gain = 0;
    for s=1:4
        r_cur  = all_results{s};
        r_next = all_results{s+1};
        % 该算法在当前阶段的排名
        cur_fits=[r_cur.best_fit]; cur_names={r_cur.name};
        idx_c=find(strcmp(cur_names, algo_names{ai_orig}));
        if isempty(idx_c), continue; end
        % 下一阶段最优值
        next_best=min([r_next.best_fit]);
        cur_val=cur_fits(idx_c);
        % 相对贡献：越接近本阶段最优，对下阶段影响越大
        stage_best_cur=min(cur_fits);
        if cur_val>0
            gain=gain+max(0,(cur_val-stage_best_cur)/cur_val)*0.25;
        end
    end
    interact_eff(ai_s) = gain;
end
interact_eff = interact_eff / max(interact_eff+1e-10) .* max(main_eff) * 0.55;

bw=0.38; x=1:nAlgo;
b1=bar(x-bw/2, main_eff,    bw,'FaceColor',C_green, 'EdgeColor','none','FaceAlpha',0.88);
b2=bar(x+bw/2, interact_eff,bw,'FaceColor',C_orange,'EdgeColor','none','FaceAlpha',0.82);

for i=1:nAlgo
    text(i-bw/2, main_eff(i)+max(main_eff)*0.012, sprintf('%.3f',main_eff(i)),...
         'HorizontalAlignment','center','FontSize',6.5,'FontWeight','bold','Color',C_green);
    text(i+bw/2, interact_eff(i)+max(main_eff)*0.012, sprintf('%.3f',interact_eff(i)),...
         'HorizontalAlignment','center','FontSize',6.5,'FontWeight','bold','Color',C_orange);
end

set(ax3,'XTick',1:nAlgo,'XTickLabel',sorted_names,'FontSize',8,...
    'GridAlpha',0.22,'TickDir','out','XTickLabelRotation',30);
legend([b1,b2],{'Main Effect (Mean |SHAP|)','Interaction (Elite Transfer Gain)'},...
       'Location','northeast','FontSize',10,'Box','on');
xlabel('Algorithm (sorted by importance)','FontWeight','bold','FontSize',11);
ylabel('Magnitude (Normalized)','FontWeight','bold','FontSize',11);
title({'图 4：主效应与交互效应对比图','All Algorithms: Main vs Interaction Effect'},...
      'FontSize',12,'FontWeight','bold');
ylim([0, max([main_eff;interact_eff])*1.22]);

saveas(fig3,'fig3_main_vs_interaction.png');
fprintf('  ✓ fig3_main_vs_interaction.png\n\n');

%% ============================================================
%  Fig 4 — SHAP 热力图（算法 × 阶段）
%  行：算法（按重要性排序）
%  列：5个阶段
%  顶部：各阶段最优值曲线 f(x)
%% ============================================================
fprintf('[Fig 4] SHAP 热力图...\n');

fig4 = figure('Color',C_bg,'Position',[120,120,1000,600],'Name','Fig4');

% 顶部 f(x) 曲线：各阶段最优值（归一化）
ax4T = subplot(6,1,1);
fx_vals  = winner_vals;
fx_norm  = (max(fx_vals)-fx_vals)/(max(fx_vals)-min(fx_vals)+1e-10);
area(1:5, fx_norm,'FaceColor',[0.25 0.25 0.25],'EdgeColor','none','FaceAlpha',0.85);
hold on;
plot(1:5, fx_norm, '-o','Color','w','LineWidth',1.5,'MarkerSize',5,...
     'MarkerFaceColor',[0.9 0.9 0.9],'MarkerEdgeColor','none');
set(ax4T,'XTick',1:5,'XTickLabel',stage_labels,'YTick',[],...
    'XLim',[0.5,5.5],'YLim',[0,1.3],'FontSize',8);
ylabel('f(x)','FontSize',8,'Rotation',0,'HorizontalAlignment','right');
title('图 8：SHAP 热力图（算法 × 阶段贡献）','FontSize',12,'FontWeight','bold');
box off;

% 热力图主体
ax4M = subplot(6,1,2:6);
% 构建热力矩阵（按重要性排序，归一化到 [-1,1]）
heat_data = sorted_shap;   % [nAlgo × 5]，已归一化到 [0,1]
heat_data_centered = heat_data * 2 - 1;  % 转为 [-1,1]

imagesc(heat_data_centered);
colormap(ax4M, cmap_go(256));
cb4 = colorbar('eastoutside');
cb4.Label.String = 'SHAP value (Green=Better contribution, Orange=Weaker)';
cb4.Label.FontSize = 8;
clim([-1 1]);

% 行标签（算法名）
set(ax4M,'YTick',1:nAlgo,'YTickLabel',sorted_names,'FontSize',8,'TickDir','out',...
    'XTick',1:5,'XTickLabel',stage_labels,'FontSize',9);
xlabel('Pipeline Stage','FontWeight','bold','FontSize',10);

% 在格子中标注排名
for ai=1:nAlgo
    for s=1:5
        if rank_mat(imp_order(ai),s)>0
            rk=rank_mat(imp_order(ai),s);
            mk=''; if rk==1, mk='★'; end
            text(s, ai, sprintf('%d%s',rk,mk),...
                 'HorizontalAlignment','center','VerticalAlignment','middle',...
                 'FontSize',7,'Color','k','FontWeight','bold');
        end
    end
end

% 阶段分界线
for s=1:4
    hold on; plot([s+0.5,s+0.5],[0.5,nAlgo+0.5],'w-','LineWidth',1.5);
end

saveas(fig4,'fig4_shap_heatmap.png');
fprintf('  ✓ fig4_shap_heatmap.png\n\n');

%% ============================================================
%  Fig 5 — 算法交互矩阵（简化版 nAlgo×nAlgo）
%  对角：各阶段综合得分分布（迷你条形）
%  上三角：两两算法SHAP交互强度散点
%  下三角：交互强度数值+条形
%% ============================================================
fprintf('[Fig 5] 算法交互矩阵...\n');

fig5 = figure('Color',C_bg,'Position',[140,140,1100,1000],'Name','Fig5');

% 为了可读性，只展示 top-10 算法（按重要性）
nShow = min(10, nAlgo);
show_idx   = imp_order(1:nShow);
show_names = algo_names(show_idx);
show_shap  = shap_mat(show_idx,:);

% 交互强度矩阵：算法 i 和 j 的协同效应（SHAP协方差）
int_strength = zeros(nShow, nShow);
for i=1:nShow
    for j=1:nShow
        vi=show_shap(i,:); vj=show_shap(j,:);
        valid=~isnan(vi)&~isnan(vj);
        if sum(valid)>1
            C=cov(vi(valid),vj(valid));
            int_strength(i,j)=abs(C(1,2));
        end
    end
end
int_norm = int_strength / max(int_strength(:)+1e-10);

for r=1:nShow
    for c=1:nShow
        ax_rc = subplot(nShow, nShow, (r-1)*nShow+c);
        hold on;
        fc = algo_color_palette(show_idx(r),:);

        if r==c
            % 对角：各阶段得分迷你条形图
            sv=show_shap(r,:);
            for ss=1:5
                if ~isnan(sv(ss))
                    bar(ss,sv(ss),0.7,'FaceColor',fc,'EdgeColor','none','FaceAlpha',0.85);
                end
            end
            ylim([0,1.1]); xlim([0.5,5.5]);
            text(3, 1.05, sprintf('%.2f',nanmean(sv)),...
                 'HorizontalAlignment','center','FontSize',6.5,'FontWeight','bold','Color',fc);

        elseif r<c
            % 上三角：两算法SHAP散点图
            vi=show_shap(r,:); vj=show_shap(c,:);
            valid=~isnan(vi)&~isnan(vj);
            if sum(valid)>0
                scatter(vi(valid),vj(valid),30,1:sum(valid),'filled','MarkerFaceAlpha',0.8);
                colormap(ax_rc, cmap_go(32));
            end
            text(0.5,0.85,sprintf('%.3f',int_norm(r,c)),...
                 'HorizontalAlignment','center','FontSize',6.5,'FontWeight','bold','Color',[0.3 0.3 0.3],...
                 'Units','normalized');
            xlim([0,1]); ylim([0,1]);

        else
            % 下三角：交互强度条形
            barh(1, int_norm(r,c), 0.55,'FaceColor',fc,'EdgeColor','none','FaceAlpha',0.80);
            text(int_norm(r,c)/2+0.01, 1, sprintf('%.3f',int_norm(r,c)),...
                 'HorizontalAlignment','center','FontSize',6.5,'FontWeight','bold','Color','w');
            xlim([0,1.15]); ylim([0.3,1.7]);
        end

        set(ax_rc,'XTick',[],'YTick',[],'Box','on',...
            'LineWidth',0.4,'XColor',[0.75 0.75 0.75],'YColor',[0.75 0.75 0.75]);
        if r==1, title(show_names{c},'FontSize',7.5,'FontWeight','bold'); end
        if c==1, ylabel(show_names{r},'FontSize',7.5,'FontWeight','bold','Rotation',45,...
                         'HorizontalAlignment','right'); end
    end
end

sgtitle({'图 5：算法交互效应复合矩阵（Top-10 算法）',...
         'SHAP Interaction Value — Algorithm × Algorithm'},...
        'FontSize',12,'FontWeight','bold');
saveas(fig5,'fig5_interaction_matrix.png');
fprintf('  ✓ fig5_interaction_matrix.png\n\n');

%% ============================================================
%  Fig 6 — 单算法 Lowess 依赖图（Top-6算法）
%  X轴：阶段进度（1~5）；Y轴：SHAP值
%  曲线：Lowess拟合；散点：各阶段真实SHAP，着色=性能值
%% ============================================================
fprintf('[Fig 6] 单算法 Lowess 依赖图...\n');

fig6 = figure('Color',C_bg,'Position',[160,160,1000,800],'Name','Fig6');

nShow6 = min(6, nAlgo);
show6  = imp_order(1:nShow6);

for pi=1:nShow6
    ax6 = subplot(3,2,pi);
    hold on; box on;
    ai  = show6(pi);
    sv  = shap_mat(ai,:);
    pv  = perf_mat(ai,:);
    fc  = algo_color_palette(ai,:);
    aname = algo_names{ai};

    valid = ~isnan(sv);
    x_v   = find(valid);
    y_v   = sv(valid);
    p_v   = pv(valid);

    if length(x_v)<2
        text(0.5,0.5,'数据不足','Units','normalized','HorizontalAlignment','center');
        title(aname,'FontSize',10,'FontWeight','bold'); continue;
    end

    % 正负着色背景
    y_pos = y_v(y_v >= 0.5); x_pos = x_v(y_v >= 0.5);
    y_neg = y_v(y_v <  0.5); x_neg = x_v(y_v <  0.5);

    % 散点（着色=性能归一化）
    p_norm=(p_v-min(p_v))/(max(p_v)-min(p_v)+1e-10);
    scatter(x_v, y_v, 55, p_norm,'filled','MarkerEdgeColor','w','LineWidth',0.8,'MarkerFaceAlpha',0.85);
    colormap(ax6, cmap_go(64));

    % Lowess 平滑（点数>=3才做）
    if length(x_v)>=3
        x_fine=linspace(min(x_v),max(x_v),100);
        try
            y_smooth=smooth(x_v,y_v,0.8,'lowess');
            y_fine=interp1(x_v,y_smooth,x_fine,'linear','extrap');
        catch
            y_fine=interp1(x_v,y_v,x_fine,'linear','extrap');
        end
        plot(x_fine, y_fine,'-','Color',fc,'LineWidth',2.5,'DisplayName','Lowess');
    end

    % 阈值线（0.5=中位）
    yline(0.5,'--','Color',[0.6 0.6 0.6],'LineWidth',1.2);

    % 正负区域填充
    if ~isempty(x_pos)
        for xi=1:length(x_pos)
            fill([x_pos(xi)-0.45,x_pos(xi)+0.45,x_pos(xi)+0.45,x_pos(xi)-0.45],...
                 [0.5,0.5,y_pos(xi),y_pos(xi)],C_green,'FaceAlpha',0.08,'EdgeColor','none');
        end
    end
    if ~isempty(x_neg)
        for xi=1:length(x_neg)
            fill([x_neg(xi)-0.45,x_neg(xi)+0.45,x_neg(xi)+0.45,x_neg(xi)-0.45],...
                 [y_neg(xi),y_neg(xi),0.5,0.5],C_orange,'FaceAlpha',0.10,'EdgeColor','none');
        end
    end

    % 最优阶段标注
    [~,best_s]=max(y_v);
    text(x_v(best_s), y_v(best_s)+0.04, sprintf('Best\nS%d',x_v(best_s)),...
         'HorizontalAlignment','center','FontSize',7,'Color',fc,'FontWeight','bold');

    set(ax6,'XTick',1:5,'XTickLabel',stage_labels,'FontSize',8,'GridAlpha',0.2,'TickDir','out');
    xlabel('Pipeline Stage','FontSize',8); ylabel('SHAP Value','FontSize',9);
    title(sprintf('[%d] %s  (Importance: %.3f)', pi, aname, algo_importance(ai)),...
          'FontSize',9.5,'FontWeight','bold','Color',fc);
    xlim([0.3,5.7]); ylim([-0.05,1.15]);
    grid on;
end

sgtitle('图 3：Top-6 算法 SHAP 单特征 Lowess 依赖图','FontSize',13,'FontWeight','bold');
saveas(fig6,'fig6_shap_dependence.png');
fprintf('  ✓ fig6_shap_dependence.png\n\n');

%% ============================================================
%  Fig 7 — 雷达图（各算法多维能力评分）
%  展示Top-5算法在6个维度的雷达对比
%% ============================================================
fprintf('[Fig 7] 雷达图...\n');

fig7 = figure('Color',C_bg,'Position',[180,180,820,760],'Name','Fig7');
ax7  = axes; hold on; box off; axis off;

% 6个评价维度
dim_names = {'Global\nSearch','Convergence','Stability','Early\nStage','Mid\nStage','Late\nStage'};
n_dim = 6;
nShow7 = min(5, nAlgo);
show7  = imp_order(1:nShow7);

% 构建各维度得分
scores7 = zeros(nShow7, n_dim);
for ki=1:nShow7
    ai=show7(ki);
    sv=shap_mat(ai,:);
    pv=perf_mat(ai,:);
    valid=~isnan(sv);

    % 维度1：全局搜索能力（Stage1 SHAP）
    scores7(ki,1) = shap_mat(ai,1);
    % 维度2：收敛速度（后期阶段改进率）
    if sum(valid)>1
        scores7(ki,2) = nanmean(sv(3:5));
    end
    % 维度3：稳定性（SHAP方差的倒数归一化）
    if sum(valid)>1
        scores7(ki,3) = max(0, 1-std(sv(valid))/0.5);
    end
    % 维度4：早期阶段（Stage 1-2 平均）
    scores7(ki,4) = nanmean(sv(1:2));
    % 维度5：中期阶段（Stage 3 或平均2-4）
    scores7(ki,5) = nanmean(sv(min(2,5):min(4,5)));
    % 维度6：后期阶段（Stage 4-5 平均）
    scores7(ki,6) = nanmean(sv(4:5));
end
% 归一化到 [0,1]
for d=1:n_dim
    col=scores7(:,d);
    scores7(:,d)=(col-min(col))/(max(col)-min(col)+1e-10);
end

angles = linspace(0, 2*pi, n_dim+1); angles=angles(1:end-1);
angles_plot=[angles, angles(1)];

% 辅助网格
for rl=[0.25,0.5,0.75,1.0]
    xc=rl*cos(angles); yc=rl*sin(angles);
    fill([xc,xc(1)],[yc,yc(1)],'none','EdgeColor',[0.85 0.85 0.85],'LineWidth',0.8);
    text(rl*cos(pi/2+0.12), rl*sin(pi/2+0.12), sprintf('%.2f',rl),...
         'FontSize',7,'Color',[0.65 0.65 0.65]);
end
for i=1:n_dim
    plot([0,cos(angles(i))],[0,sin(angles(i))],'-','Color',[0.85 0.85 0.85],'LineWidth',0.8);
end

% 各算法多边形
for ki=1:nShow7
    ai  = show7(ki);
    fc  = algo_color_palette(ai,:);
    aname=algo_names{ai};
    sv  = [scores7(ki,:), scores7(ki,1)];
    xp  = sv .* cos(angles_plot);
    yp  = sv .* sin(angles_plot);
    fill(xp, yp, fc,'FaceAlpha',0.08,'EdgeColor','none');
    plot(xp, yp,'-o','Color',fc,'LineWidth',2.0,'MarkerSize',6,...
         'MarkerFaceColor',fc,'MarkerEdgeColor','w',...
         'DisplayName',sprintf('%s (%.3f)',aname,algo_importance(ai)));
end

% 维度标签
dim_labels_clean = {'Global Search','Convergence','Stability','Early Stage','Mid Stage','Late Stage'};
for i=1:n_dim
    lx=1.22*cos(angles(i)); ly=1.22*sin(angles(i));
    text(lx, ly, dim_labels_clean{i},'HorizontalAlignment','center',...
         'FontSize',9,'FontWeight','bold','Color',[0.2 0.2 0.2]);
end

legend('Location','southoutside','Orientation','horizontal','FontSize',9,...
       'Box','off','NumColumns',3);
title({'图 7：Top-5 算法综合能力雷达图',...
       'Multi-dimensional Performance Comparison'},...
      'FontSize',12,'FontWeight','bold','Units','normalized','Position',[0.5,0.97]);
axis equal; axis([-1.5,1.5,-1.4,1.55]);

% 右下角：全局重要性排名简表
info_lines = '';
for ki=1:nShow7
    ai=show7(ki);
    info_lines=[info_lines, sprintf('#%d %s:%.3f  ',ki,algo_names{ai},algo_importance(ai))];
end
text(1.45,-1.35,info_lines,'FontSize',7,'Color',[0.45 0.45 0.45],...
     'HorizontalAlignment','right','Units','data');

saveas(fig7,'fig7_radar.png');
fprintf('  ✓ fig7_radar.png\n\n');

%% ==================== 完成汇总 ====================
fprintf('========================================================\n');
fprintf('  全部完成！共生成图表：\n');
fprintf('  [竞选原图]  各阶段竞选结果（原始分析图）\n');
fprintf('  [SHAP评估]\n');
fprintf('    fig1_election_ranking.png\n');
fprintf('    fig2_shap_beeswarm.png\n');
fprintf('    fig3_main_vs_interaction.png\n');
fprintf('    fig4_shap_heatmap.png\n');
fprintf('    fig5_interaction_matrix.png\n');
fprintf('    fig6_shap_dependence.png\n');
fprintf('    fig7_radar.png\n');
fprintf('\n  全局重要性排名（Top-5）：\n');
for ki=1:min(5,nAlgo)
    ai=imp_order(ki);
    fprintf('    #%d  %-6s  Importance=%.4f\n',ki,algo_names{ai},algo_importance(ai));
end
fprintf('\n  总运行时间：%.2f 秒\n', toc);
fprintf('========================================================\n');

%% ============================================================
%  子函数：竞选框架
%% ============================================================

function results = run_stage_competition(fobj, dim, Lb, Ub, NFE, elite_pop, elite_fit, K_elite)
    idx=0;
    results=struct('name',{},'best_fit',{},'best_pos',{},'pop',{},'fit',{});
    algo_list={'GA',@run_GA;'DE',@run_DE;'PSO',@run_PSO;'SSA',@run_SSA;...
               'GWO',@run_GWO;'FA',@run_FA;'ABC',@run_ABC;'TOW',@run_TOW;...
               'CA',@run_CA;'PO',@run_PO;'CS',@run_CS;'HLO',@run_HLO;...
               'SA',@run_SA;'HS',@run_HS;'NSGA',@run_NSGA};
    for a=1:size(algo_list,1)
        name=algo_list{a,1}; fn=algo_list{a,2};
        fprintf('  评测 %-4s ... ',name);
        try
            [bf,bp,pop_out,fit_out]=fn(fobj,dim,Lb,Ub,NFE,elite_pop,elite_fit,K_elite);
            idx=idx+1;
            results(idx).name=name; results(idx).best_fit=bf;
            results(idx).best_pos=bp; results(idx).pop=pop_out; results(idx).fit=fit_out;
            fprintf('最优: %.2f\n',bf);
        catch ME
            fprintf('[错误跳过] %s\n',ME.message);
        end
    end
end

function print_ranking(results)
    [fits_s,si]=sort([results.best_fit]);
    fprintf('  排名 | 算法  | 最优适应度\n');
    fprintf('  -----|-------|--------------------\n');
    for r=1:length(si)
        mk=''; if r==1, mk=' ← 胜出'; end
        fprintf('  %3d  | %-4s  | %.2f m%s\n',r,results(si(r)).name,fits_s(r),mk);
    end
end

function [ep,ef]=get_elite_pool(results,K,dim)
    all_pop=[]; all_fit=[];
    for i=1:length(results)
        if ~isempty(results(i).pop)&&~isempty(results(i).fit)
            all_pop=[all_pop;results(i).pop];
            all_fit=[all_fit;results(i).fit(:)];
        end
    end
    if isempty(all_pop), ep=[]; ef=[]; return; end
    [~,si]=sort(all_fit); K=min(K,length(si));
    ep=all_pop(si(1:K),:); ef=all_fit(si(1:K));
end

function pop=init_pop(popSize,dim,Lb,Ub,elite_pop,K_elite)
    pop=zeros(popSize,dim); ne=min(size(elite_pop,1),K_elite);
    for i=1:popSize
        if i<=ne&&ne>0, pop(i,:)=elite_pop(i,:);
        elseif ne>0
            base=elite_pop(randi(ne),:);
            noise=round(randn(1,dim).*max(1,(Ub-Lb)*0.04));
            pop(i,:)=max(Lb,min(Ub,base+noise));
        else, pop(i,:)=Lb+round(rand(1,dim).*(Ub-Lb)); end
    end
end

%% ============================================================
%  15个算法实现
%% ============================================================

function [best_fit,best_pos,pop,fit]=run_GA(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    popSize=60; maxGen=floor(NFE/popSize); pc=0.85; pm=0.12;
    pop=init_pop(popSize,dim,Lb,Ub,ep,K);
    fit=arrayfun(@(i) fobj(pop(i,:)),1:popSize)';
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);
    for iter_=1:maxGen
        newP=zeros(size(pop));
        for i=1:popSize
            c=randperm(popSize,3); [~,w]=min(fit(c)); newP(i,:)=pop(c(w),:);
        end
        for i=1:2:popSize-1
            if rand<pc
                cp=randi(dim); tmp=newP(i,cp:end);
                newP(i,cp:end)=newP(i+1,cp:end); newP(i+1,cp:end)=tmp;
            end
        end
        for i=1:popSize
            for j=1:dim
                if rand<pm, newP(i,j)=Lb(j)+randi(Ub(j)-Lb(j)+1)-1; end
            end
        end
        newP=max(Lb,min(Ub,newP)); pop=newP;
        for i=1:popSize
            fit(i)=fobj(pop(i,:));
            if fit(i)<best_fit, best_fit=fit(i); best_pos=pop(i,:); end
        end
        pop(1,:)=best_pos; fit(1)=best_fit;
    end
end

function [best_fit,best_pos,pop,fit]=run_DE(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    popSize=50; maxGen=floor(NFE/popSize); F=0.7; CR=0.8;
    pop=init_pop(popSize,dim,Lb,Ub,ep,K);
    fit=arrayfun(@(i) fobj(pop(i,:)),1:popSize)';
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);
    for iter_=1:maxGen
        for i=1:popSize
            cands=setdiff(1:popSize,i); r=cands(randperm(length(cands),3));
            v=max(Lb,min(Ub,round(pop(r(1),:)+F*(pop(r(2),:)-pop(r(3),:)))));
            mask=rand(1,dim)<CR; mask(randi(dim))=true;
            trial=pop(i,:); trial(mask)=v(mask);
            trial=max(Lb,min(Ub,trial)); ft=fobj(trial);
            if ft<fit(i), pop(i,:)=trial; fit(i)=ft; end
            if ft<best_fit, best_fit=ft; best_pos=trial; end
        end
    end
end

function [best_fit,best_pos,pop,fit]=run_PSO(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    popSize=50; maxIter=floor(NFE/popSize); w=0.8; c1=1.5; c2=1.5;
    pop=init_pop(popSize,dim,Lb,Ub,ep,K); vel=zeros(popSize,dim);
    fit=arrayfun(@(i) fobj(pop(i,:)),1:popSize)';
    pbest=pop; pbest_fit=fit;
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);
    for iter_=1:maxIter
        for i=1:popSize
            vel(i,:)=w*vel(i,:)+c1*rand*(pbest(i,:)-pop(i,:))+c2*rand*(best_pos-pop(i,:));
            pop(i,:)=max(Lb,min(Ub,round(pop(i,:)+vel(i,:))));
            f=fobj(pop(i,:)); fit(i)=f;
            if f<pbest_fit(i), pbest(i,:)=pop(i,:); pbest_fit(i)=f; end
            if f<best_fit, best_fit=f; best_pos=pop(i,:); end
        end
    end
end

function [best_fit,best_pos,pop,fit]=run_SSA(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    popSize=50; maxIter=floor(NFE/popSize); P_pct=0.2;
    pop=double(init_pop(popSize,dim,Lb,Ub,ep,K));
    fit=arrayfun(@(i) fobj(pop(i,:)),1:popSize)';
    pFit=fit; pX=pop;
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);
    for t=1:maxIter
        [~,sortIdx]=sort(fit); [~,wIdx]=max(fit);
        worse=pop(wIdx,:); nd=round(popSize*P_pct);
        for i=1:nd
            si=sortIdx(i);
            if rand<0.8, pop(si,:)=round(pX(si,:).*exp(-i/(rand*maxIter)));
            else, pop(si,:)=round(pX(si,:)+randn(1,dim)); end
        end
        for i=nd+1:popSize
            si=sortIdx(i);
            if i>popSize/2
                pop(si,:)=round(randn(1,dim).*exp((worse-pX(si,:))./(i^2)));
            else
                A=sign(rand(1,dim)-0.5);
                pop(si,:)=round(best_pos+abs(pX(si,:)-best_pos).*A);
            end
        end
        for i=1:popSize
            pop(i,:)=max(Lb,min(Ub,pop(i,:))); fit(i)=fobj(pop(i,:));
            if fit(i)<pFit(i), pFit(i)=fit(i); pX(i,:)=pop(i,:); end
            if pFit(i)<best_fit, best_fit=pFit(i); best_pos=pX(i,:); end
        end
    end
end

function [best_fit,best_pos,pop,fit]=run_GWO(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    n=50; maxIter=floor(NFE/n);
    pop=init_pop(n,dim,Lb,Ub,ep,K);
    fit=arrayfun(@(i) fobj(pop(i,:)),1:n)';
    [~,si]=sort(fit);
    Ap=pop(si(1),:); As=fit(si(1));
    Bp=pop(si(2),:); Bs=fit(si(2));
    Dp=pop(si(3),:); Ds=fit(si(3));
    best_fit=As; best_pos=Ap;
    for l=1:maxIter
        a=2-l*(2/maxIter);
        for i=1:n
            np=zeros(1,dim);
            for j=1:dim
                X1=Ap(j)-(2*a*rand-a)*abs(2*rand*Ap(j)-pop(i,j));
                X2=Bp(j)-(2*a*rand-a)*abs(2*rand*Bp(j)-pop(i,j));
                X3=Dp(j)-(2*a*rand-a)*abs(2*rand*Dp(j)-pop(i,j));
                np(j)=round((X1+X2+X3)/3);
            end
            np=max(Lb,min(Ub,np)); f=fobj(np);
            pop(i,:)=np; fit(i)=f;
            if f<As, Ds=Bs;Dp=Bp; Bs=As;Bp=Ap; As=f;Ap=np;
            elseif f<Bs, Ds=Bs;Dp=Bp; Bs=f;Bp=np;
            elseif f<Ds, Ds=f;Dp=np; end
        end
        if As<best_fit, best_fit=As; best_pos=Ap; end
    end
end

function [best_fit,best_pos,pop,fit]=run_FA(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    n=40; maxIter=floor(NFE/n);
    alpha=0.5; betamin=0.2; gamma=1;
    pop=double(init_pop(n,dim,Lb,Ub,ep,K));
    fit=arrayfun(@(i) fobj(pop(i,:)),1:n)';
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);
    for iter_=1:maxIter
        [fit,idx]=sort(fit); pop=pop(idx,:);
        for i=1:n
            for j=1:n
                if fit(i)>fit(j)
                    r=norm(pop(i,:)-pop(j,:));
                    beta=(1-betamin)*exp(-gamma*r^2)+betamin;
                    pop(i,:)=round(pop(i,:)*(1-beta)+pop(j,:)*beta+alpha*(rand(1,dim)-0.5));
                    pop(i,:)=max(Lb,min(Ub,pop(i,:))); fit(i)=fobj(pop(i,:));
                end
            end
        end
        [fmin,bi]=min(fit);
        if fmin<best_fit, best_fit=fmin; best_pos=pop(bi,:); end
    end
end

function [best_fit,best_pos,pop,fit]=run_ABC(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    FN=25; limit=20; maxCycle=floor(NFE/(FN*2));
    pop=init_pop(FN,dim,Lb,Ub,ep,K);
    fit=arrayfun(@(i) fobj(pop(i,:)),1:FN)';
    trial=zeros(1,FN);
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);
    for iter_=1:maxCycle
        for i=1:FN
            k=i; while k==i, k=randi(FN); end
            phi=randi([-1,1],1,dim);
            nf=max(Lb,min(Ub,round(pop(i,:)+phi.*(pop(i,:)-pop(k,:)))));
            fn=fobj(nf);
            if fn<fit(i), pop(i,:)=nf; fit(i)=fn; trial(i)=0;
            else, trial(i)=trial(i)+1; end
            if fn<best_fit, best_fit=fn; best_pos=nf; end
        end
        for i=1:FN
            if trial(i)>limit
                if size(ep,1)>0
                    base=ep(randi(size(ep,1)),:);
                    pop(i,:)=max(Lb,min(Ub,base+round(randn(1,dim)*2)));
                else
                    pop(i,:)=Lb+round(rand(1,dim).*(Ub-Lb));
                end
                fit(i)=fobj(pop(i,:)); trial(i)=0;
                if fit(i)<best_fit, best_fit=fit(i); best_pos=pop(i,:); end
            end
        end
    end
end

function [best_fit,best_pos,pop,fit]=run_TOW(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    nT=30; maxIter=floor(NFE/nT); alpha_t=0.98; sigma0=2.0;
    pop=double(init_pop(nT,dim,Lb,Ub,ep,K));
    fit=arrayfun(@(i) fobj(pop(i,:)),1:nT)';
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);
    for it=1:maxIter
        inv_f=1./(fit-min(fit)+1e-10); W=inv_f/sum(inv_f); wc=W'*pop;
        sigma=sigma0*alpha_t^it;
        for i=1:nT
            step=round(0.6*(wc-pop(i,:))+sigma*randn(1,dim));
            np=max(Lb,min(Ub,pop(i,:)+step)); f=fobj(np);
            if f<fit(i), pop(i,:)=np; fit(i)=f; end
            if f<best_fit, best_fit=f; best_pos=np; end
        end
    end
end

function [best_fit,best_pos,pop,fit]=run_CA(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    nPop=50; nAccept=round(0.2*nPop); alpha_ca=0.15;
    maxIter=floor(NFE/nPop);
    pop=double(init_pop(nPop,dim,Lb,Ub,ep,K));
    fit=arrayfun(@(i) fobj(pop(i,:)),1:nPop)';
    [~,si]=sort(fit); best_fit=fit(si(1)); best_pos=pop(si(1),:);
    cult_best=best_pos; norm_lo=Lb; norm_hi=Ub;
    for iter_=1:maxIter
        for i=1:nPop
            sigma=alpha_ca*(norm_hi-norm_lo);
            dx=round(sigma.*randn(1,dim))+round(0.3*sign(cult_best-pop(i,:)));
            pop(i,:)=max(Lb,min(Ub,round(pop(i,:)+dx))); fit(i)=fobj(pop(i,:));
        end
        [~,si]=sort(fit); spop=pop(si(1:nAccept),:);
        norm_lo=min(spop,[],1); norm_hi=max(spop,[],1)+1;
        if fit(si(1))<best_fit
            best_fit=fit(si(1)); best_pos=pop(si(1),:); cult_best=best_pos;
        end
    end
end

function [best_fit,best_pos,pop,fit]=run_PO(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    parties=10; areas=10; popSize=parties*areas; maxIter=floor(NFE/popSize);
    pop=init_pop(popSize,dim,Lb,Ub,ep,K);
    fit=arrayfun(@(i) fobj(pop(i,:)),1:popSize)';
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);
    for it=1:maxIter
        for p=1:parties
            idx_p=(p-1)*areas+1:p*areas; [~,ll]=min(fit(idx_p));
            leader_idx=idx_p(ll); leader=pop(leader_idx,:);
            for m=idx_p
                if m==leader_idx, continue; end
                lr=0.3+0.4*rand;
                step=round((leader-pop(m,:))*lr+randn(1,dim)*0.8);
                np=max(Lb,min(Ub,pop(m,:)+step)); f=fobj(np);
                if f<fit(m), pop(m,:)=np; fit(m)=f; end
                if f<best_fit, best_fit=f; best_pos=np; end
            end
        end
        [~,gsi]=sort(fit); global_leader=pop(gsi(1),:);
        for p=1:parties
            idx_p=(p-1)*areas+1:p*areas; [~,ll]=min(fit(idx_p)); li=idx_p(ll);
            step=round((global_leader-pop(li,:))*0.25*rand+randn(1,dim)*0.6);
            np=max(Lb,min(Ub,pop(li,:)+step)); f=fobj(np);
            if f<fit(li), pop(li,:)=np; fit(li)=f; end
            if f<best_fit, best_fit=f; best_pos=np; end
        end
    end
end

function [best_fit,best_pos,pop,fit]=run_CS(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    n=30; maxIter=floor(NFE/n); pa=0.25; beta=1.5;
    pop=init_pop(n,dim,Lb,Ub,ep,K);
    fit=arrayfun(@(i) fobj(pop(i,:)),1:n)';
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);
    for iter_=1:maxIter
        for i=1:n
            levy=levy_cs(beta,dim);
            scale=max(1,round(0.05*mean(Ub-Lb)));
            np=max(Lb,min(Ub,round(pop(i,:)+scale*levy.*(pop(i,:)-best_pos)+randn(1,dim))));
            f=fobj(np); j=randi(n);
            if f<fit(j), pop(j,:)=np; fit(j)=f; end
            if f<best_fit, best_fit=f; best_pos=np; end
        end
        for i=1:n
            if rand<pa
                if size(ep,1)>0
                    np=max(Lb,min(Ub,ep(randi(size(ep,1)),:)+randi([-10,10],1,dim)));
                else, np=Lb+round(rand(1,dim).*(Ub-Lb)); end
                fit(i)=fobj(np); pop(i,:)=np;
                if fit(i)<best_fit, best_fit=fit(i); best_pos=np; end
            end
        end
    end
end

function levy=levy_cs(beta,dim)
    sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u=randn(1,dim)*sigma; v=randn(1,dim);
    levy=u./(abs(v).^(1/beta)); levy=sign(levy).*min(abs(levy),5);
end

function [best_fit,best_pos,pop,fit]=run_HLO(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    popSize=40; bpv=8; m=dim*bpv; maxIter=floor(NFE/popSize);
    p_r=0.1; p_i=0.5;
    bin_pop=zeros(popSize,m);
    for i=1:popSize
        if i<=min(size(ep,1),K)&&size(ep,1)>0
            for v=1:dim
                ratio=(double(ep(i,v))-Lb(v))/(max(Ub(v)-Lb(v),1));
                int_val=round(ratio*(2^bpv-1));
                bits=dec2bin(int_val,bpv)-'0';
                bin_pop(i,(v-1)*bpv+1:v*bpv)=bits;
            end
        else, bin_pop(i,:)=randi([0,1],1,m); end
    end
    decode=@(row) arrayfun(@(v) max(Lb(v),min(Ub(v),Lb(v)+round(...
        sum(row((v-1)*bpv+1:v*bpv).*(2.^(bpv-1:-1:0)))/(2^bpv-1)*(Ub(v)-Lb(v))))),1:dim);
    IKD=bin_pop;
    IKDfits=arrayfun(@(i) fobj(decode(bin_pop(i,:))),1:popSize)';
    [best_val,bi]=min(IKDfits); SKD=IKD(bi,:); SKDfit=best_val;
    best_pos=decode(SKD); best_fit=SKDfit;
    pop_int=zeros(popSize,dim);
    for i=1:popSize, pop_int(i,:)=decode(bin_pop(i,:)); end
    for iter_=1:maxIter
        for i=1:popSize
            for j=1:m
                pr=rand;
                if pr<p_r, bin_pop(i,j)=randi([0,1]);
                elseif pr<p_i, bin_pop(i,j)=IKD(i,j);
                else, bin_pop(i,j)=SKD(j); end
            end
            x_int=decode(bin_pop(i,:)); fv=fobj(x_int);
            if fv<IKDfits(i), IKDfits(i)=fv; IKD(i,:)=bin_pop(i,:); end
            if fv<SKDfit, SKDfit=fv; SKD=bin_pop(i,:); best_pos=x_int; best_fit=fv; end
        end
        for i=1:popSize, pop_int(i,:)=decode(bin_pop(i,:)); end
    end
    pop=pop_int; fit=IKDfits;
end

function [best_fit,best_pos,pop,fit]=run_SA(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    try
        DFenPei_=evalin('base','DFenPei'); dis_mat_=evalin('base','data.dis');
    catch
        pop=init_pop(20,dim,Lb,Ub,ep,K);
        fit=arrayfun(@(i) fobj(pop(i,:)),1:20)';
        [best_fit,bi]=min(fit); best_pos=pop(bi,:); return;
    end
    T0=1e5; Tend=1; q=0.93; L=30; NFE_used=0;
    all_cands=unique(cell2mat(cellfun(@(c) c(2:end),DFenPei_,'UniformOutput',false)));
    nCands=length(all_cands); num_sel=min(57,nCands);
    S1=all_cands(randperm(nCands,num_sel));
    x_int1=sa_to_fobj_x(S1,DFenPei_,Lb,Ub);
    best_fit=fobj(x_int1); NFE_used=NFE_used+1;
    best_S=S1; best_pos=x_int1;
    while T0>Tend&&NFE_used<NFE
        for i=1:L
            S2=S1; unsel=setdiff(all_cands,S1);
            if isempty(unsel), break; end
            S2(randi(num_sel))=unsel(randi(length(unsel)));
            x_int2=sa_to_fobj_x(S2,DFenPei_,Lb,Ub);
            f2=fobj(x_int2); NFE_used=NFE_used+1;
            delta=f2-best_fit;
            if f2<best_fit||exp(-delta/T0)>rand
                S1=S2;
                if f2<best_fit, best_fit=f2; best_S=S2; best_pos=x_int2; end
            end
            if NFE_used>=NFE, break; end
        end
        T0=T0*q;
    end
    pop_size=min(20,dim); pop=repmat(best_pos,pop_size,1);
    for i=2:pop_size, pop(i,:)=max(Lb,min(Ub,best_pos+round(randn(1,dim)))); end
    fit=arrayfun(@(i) fobj(pop(i,:)),1:pop_size)';
end

function x_int=sa_to_fobj_x(sel_centers,DFenPei_,Lb,Ub)
    dim_=length(DFenPei_); x_int=zeros(1,dim_);
    for i=1:dim_
        cands=DFenPei_{i}(2:end); overlap=intersect(cands,sel_centers);
        if ~isempty(overlap), idx=find(DFenPei_{i}==overlap(1))-1;
        else, idx=1; end
        x_int(i)=max(Lb(i),min(Ub(i),idx));
    end
end

function [best_fit,best_pos,pop,fit]=run_HS(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    try
        DFenPei_=evalin('base','DFenPei'); data_=evalin('base','data');
        binan_xy=data_.binan; start_xy=data_.start;
    catch
        pop=init_pop(20,dim,Lb,Ub,ep,K);
        fit=arrayfun(@(i) fobj(pop(i,:)),1:20)';
        [best_fit,bi]=min(fit); best_pos=pop(bi,:); return;
    end
    HMS=15; num_centers=min(57,size(binan_xy,1));
    house_x=start_xy(:,1); house_y=start_xy(:,2);
    min_x=min(house_x); max_x=max(house_x); min_y=min(house_y); max_y=max(house_y);
    NVAR=num_centers*2; NFE_used=0; maxItr=floor(NFE/HMS);
    BW_max=(max_x-min_x)*0.2; BW_min=(max_x-min_x)*0.0001;
    HM=zeros(HMS,NVAR); hm_fit=zeros(HMS,1);
    for i=1:HMS
        idx_r=randperm(size(house_x,1),num_centers);
        pos=[house_x(idx_r),house_y(idx_r)]; HM(i,:)=pos(:)';
        hm_fit(i)=hs_eval(HM(i,:),num_centers,house_x,house_y,fobj,DFenPei_,binan_xy,Lb,Ub);
        NFE_used=NFE_used+1;
    end
    [best_hf,bi]=min(hm_fit); best_harm=HM(bi,:);
    for itr=1:maxItr
        if NFE_used>=NFE, break; end
        BW=BW_max*exp(log(BW_min/BW_max)*itr/maxItr);
        [~,bi_]=min(hm_fit); new_h=HM(bi_,:);
        t=randi(num_centers); ix=t*2-1; iy=t*2;
        new_h(ix)=max(min(new_h(ix)+(rand*2-1)*BW,max_x),min_x);
        new_h(iy)=max(min(new_h(iy)+(rand*2-1)*BW,max_y),min_y);
        fnew=hs_eval(new_h,num_centers,house_x,house_y,fobj,DFenPei_,binan_xy,Lb,Ub);
        NFE_used=NFE_used+1;
        [worst_f,wi]=max(hm_fit);
        if fnew<worst_f
            HM(wi,:)=new_h; hm_fit(wi)=fnew;
            if fnew<best_hf, best_hf=fnew; best_harm=new_h; end
        end
    end
    best_pos=hs_harm_to_int(best_harm,num_centers,DFenPei_,binan_xy,Lb,Ub);
    best_fit=fobj(best_pos);
    pop_size=min(20,HMS); pop=repmat(best_pos,pop_size,1);
    for i=2:pop_size, pop(i,:)=max(Lb,min(Ub,best_pos+round(randn(1,dim)))); end
    fit=arrayfun(@(i) fobj(pop(i,:)),1:pop_size)';
end

function fv=hs_eval(harm,nc,hx,hy,fobj,DFenPei_,binan_xy,Lb,Ub)
    x_int=hs_harm_to_int(harm,nc,DFenPei_,binan_xy,Lb,Ub); fv=fobj(x_int);
end

function x_int=hs_harm_to_int(harm,nc,DFenPei_,binan_xy,Lb,Ub)
    centers=reshape(harm,[],2); n_binan=size(binan_xy,1); sel=zeros(1,nc);
    for c=1:nc
        d=sqrt((binan_xy(:,1)-centers(c,1)).^2+(binan_xy(:,2)-centers(c,2)).^2);
        [~,mi]=min(d); sel(c)=mi;
    end
    sel=unique(sel); x_int=zeros(1,length(DFenPei_));
    for i=1:length(DFenPei_)
        cands=DFenPei_{i}(2:end); overlap=intersect(cands,sel);
        if ~isempty(overlap), idx=find(DFenPei_{i}==overlap(1))-1;
        else, idx=1; end
        x_int(i)=max(Lb(i),min(Ub(i),idx));
    end
end

function [best_fit,best_pos,pop,fit]=run_NSGA(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    try
        DFenPei_=evalin('base','DFenPei'); data_=evalin('base','data'); dis_mat_=data_.dis;
    catch
        pop=init_pop(20,dim,Lb,Ub,ep,K);
        fit=arrayfun(@(i) fobj(pop(i,:)),1:20)';
        [best_fit,bi]=min(fit); best_pos=pop(bi,:); return;
    end
    popSize=min(100,floor(NFE/20)); maxGen=floor(NFE/popSize);
    P_=cellfun(@(x) length(x)-1,DFenPei_);
    nsga_obj=@(x) nsga_eval(x,DFenPei_,dis_mat_,P_);
    if ~isempty(ep), pop=init_pop(popSize,dim,Lb,Ub,ep,K);
    else, pop=Lb+round(rand(popSize,dim).*(Ub-Lb)); end
    F=zeros(popSize,2);
    for i=1:popSize, F(i,:)=nsga_obj(pop(i,:)); end
    for g=1:maxGen
        child=zeros(popSize,dim);
        for i=1:popSize
            p1=pop(randi(popSize),:); p2=pop(randi(popSize),:);
            cp=randi(dim); child(i,:)=[p1(1:cp-1),p2(cp:end)];
            if rand<0.1
                j1=randi(dim); j2=randi(dim);
                child(i,j1)=Lb(j1)+randi(max(1,Ub(j1)-Lb(j1)+1))-1;
            end
            child(i,:)=max(Lb,min(Ub,child(i,:)));
        end
        combined=[pop;child]; Fc=zeros(2*popSize,2);
        for i=1:2*popSize, Fc(i,:)=nsga_obj(combined(i,:)); end
        [~,si]=sort(Fc(:,1)); pop=combined(si(1:popSize),:); F=Fc(si(1:popSize),:);
    end
    [~,bi]=min(F(:,1)); best_pos=pop(bi,:); best_fit=fobj(best_pos);
    fit=arrayfun(@(i) fobj(pop(i,:)),1:popSize)';
end

function f2=nsga_eval(x,DFenPei_,dis_mat_,P_)
    X=max(1,min(round(x),P_)); total_d=0; Y=zeros(1,size(dis_mat_,2));
    for i=1:length(X)
        hid=DFenPei_{i}(1); eid=DFenPei_{i}(X(i)+1);
        total_d=total_d+dis_mat_(hid,eid); Y(eid)=Y(eid)+12;
    end
    f2=[total_d,var(Y)];
end

%% ============================================================
%  辅助：统一目标函数 + 色图
%% ============================================================

function fitness=unified_fobj(x,DFenPei,dis_mat,Lb,Ub)
    fitness=0;
    for i=1:length(DFenPei)
        idx=max(Lb(i),min(round(x(i)),Ub(i)));
        fitness=fitness+dis_mat(DFenPei{i}(1),DFenPei{i}(idx+1));
    end
end

function cmap=cmap_go(n)
% 绿→白→橙渐变色图（SHAP参考图配色）
    c1=[0.12 0.45 0.20]; c2=[0.92 0.96 0.88]; c3=[0.85 0.55 0.10];
    h=floor(n/2);
    s1=[linspace(c1(1),c2(1),h)',linspace(c1(2),c2(2),h)',linspace(c1(3),c2(3),h)'];
    s2=[linspace(c2(1),c3(1),n-h)',linspace(c2(2),c3(2),n-h)',linspace(c2(3),c3(3),n-h)'];
    cmap=[s1;s2];
end