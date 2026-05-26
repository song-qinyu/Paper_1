%% ============================================================
%  Parallel Algorithm Benchmark — 15 Algorithms, 5 Categories
%
%  Algorithms grouped by type (matching reference image):
%    Evolutionary  : GA, DE, NSGA
%    Swarm         : PSO, SSA, GWO
%    Nature-inspired: CS, FA, ABC
%    Physics-based : SA, HS, TOW
%    Human-based   : CA, PO, HLO
%
%  Per category: select the best-performing algorithm.
%  Visualization: same green-gold SHAP style + category figures.
%% ============================================================
clc; clear; close all; tic;

fprintf('========================================================\n');
fprintf('  Parallel Algorithm Benchmark — 15 Algorithms\n');
fprintf('  5 Categories | 1 Winner per Category\n');
fprintf('========================================================\n\n');

%% ==================== 0. Data Loading ====================
if exist('sj5.mat','file'), load('sj5.mat');
else, error('sj5.mat not found'); end

if exist('dis','var'), data.dis = dis; end

dim = length(DFenPei);
Lb  = ones(1, dim);
Ub  = arrayfun(@(i) length(DFenPei{i})-1, 1:dim);

fobj = @(x) unified_fobj(x, DFenPei, data.dis, Lb, Ub);

%% ==================== 1. Category Definitions ====================
% ── 5 categories matching the reference image ─────────────────
CAT_NAMES   = {'Evolutionary','Swarm','Nature-inspired','Physics-based','Human-based'};
CAT_ALGOS   = {
    {'GA','DE','NSGA'},        % Evolutionary
    {'PSO','SSA','GWO'},       % Swarm
    {'CS','FA','ABC'},         % Nature-inspired
    {'SA','HS','TOW'},         % Physics-based
    {'CA','PO','HLO'}          % Human-based
};
nCat  = length(CAT_NAMES);

% Full ordered list (must stay consistent with all matrices below)
ALGO_NAMES = {'GA','DE','NSGA','PSO','SSA','GWO','CS','FA','ABC','SA','HS','TOW','CA','PO','HLO'};
nAlgo      = length(ALGO_NAMES);

% Category index for each algorithm
algo_cat = zeros(1, nAlgo);
for c = 1:nCat
    for a = 1:nAlgo
        if any(strcmp(CAT_ALGOS{c}, ALGO_NAMES{a}))
            algo_cat(a) = c;
        end
    end
end

%% ==================== 2. Color Palette ====================
% Green-gold base (pipeline-consistent)
C_GREEN = [88,  140,  90] /255;
C_GOLD  = [214, 164,  59] /255;
C_LG    = [160, 200, 130] /255;
C_LY    = [240, 210, 140] /255;
C_W     = [252, 251, 248] /255;
C_MID   = 0.5*C_GREEN + 0.5*C_GOLD;

% One distinct colour per category (green-gold family)
CAT_COLORS = {
    C_GREEN,                    % Evolutionary  — dark green
    [0.30 0.68 0.45],           % Swarm         — mid green
    C_MID,                      % Nature-insp   — green-gold blend
    [0.93 0.75 0.25],           % Physics-based — warm gold
    C_GOLD                      % Human-based   — deep gold
};

% Gradient colormap for heatmaps
n_c = 256; h2 = n_c/2;
cmap_gg = [ ...
    linspace(C_GREEN(1),0.97,h2)', linspace(C_GREEN(2),0.97,h2)', linspace(C_GREEN(3),0.97,h2)'; ...
    linspace(0.97,C_GOLD(1),h2)',  linspace(0.97,C_GOLD(2),h2)',  linspace(0.97,C_GOLD(3),h2)'];

%% ==================== 3. Benchmark Settings ====================
NFE_total = 75000;
N_RUNS    = 5;

fprintf('NFE per algorithm : %d\n', NFE_total);
fprintf('Independent runs  : %d\n', N_RUNS);
fprintf('Algorithms        : %d  (%d categories)\n\n', nAlgo, nCat);

%% ==================== 4. Parallel Evaluation ====================
N_CKPT   = 20;
ckpt_nfe = round(linspace(1, NFE_total, N_CKPT));

best_fit_mat = nan(nAlgo, N_RUNS);
best_pos_mat = cell(nAlgo, N_RUNS);
conv_mat     = nan(nAlgo, N_RUNS, N_CKPT);

use_parallel = ~isempty(ver('parallel'));
if use_parallel
    try
        pool = gcp('nocreate');
        if isempty(pool), parpool('local'); end
        fprintf('Parallel pool active. Running with parfor.\n\n');
    catch
        use_parallel = false;
        fprintf('Could not start parallel pool. Falling back to for-loop.\n\n');
    end
else
    fprintf('Parallel Computing Toolbox not found. Running sequentially.\n\n');
end

DFenPei_snap = DFenPei;
data_snap    = data;
Lb_snap      = Lb;
Ub_snap      = Ub;
dim_snap     = dim;

fprintf('Starting evaluation...\n');
fprintf('%-18s | %-6s | %-12s | %-12s | %-10s\n', ...
    'Category','Algo','Best','Mean','Std');
fprintf('%s\n', repmat('-',1,66));

for a = 1:nAlgo
    aname    = ALGO_NAMES{a};
    cat_name = CAT_NAMES{algo_cat(a)};

    tmp_best = nan(1, N_RUNS);
    tmp_pos  = cell(1, N_RUNS);
    tmp_conv = nan(N_RUNS, N_CKPT);

    fobj_local = @(x) unified_fobj(x, DFenPei_snap, data_snap.dis, Lb_snap, Ub_snap);

    if use_parallel
        parfor r = 1:N_RUNS
            [bf, bp, cv] = run_single_algo(aname, fobj_local, dim_snap, ...
                Lb_snap, Ub_snap, NFE_total, N_CKPT, ckpt_nfe, ...
                DFenPei_snap, data_snap);
            tmp_best(r) = bf;
            tmp_pos{r}  = bp;
            tmp_conv(r,:) = cv;
        end
    else
        for r = 1:N_RUNS
            [bf, bp, cv] = run_single_algo(aname, fobj_local, dim_snap, ...
                Lb_snap, Ub_snap, NFE_total, N_CKPT, ckpt_nfe, ...
                DFenPei_snap, data_snap);
            tmp_best(r) = bf;
            tmp_pos{r}  = bp;
            tmp_conv(r,:) = cv;
        end
    end

    best_fit_mat(a,:) = tmp_best;
    best_pos_mat(a,:) = tmp_pos;
    conv_mat(a,:,:)   = tmp_conv;

    fprintf('%-18s | %-6s | %-12.2f | %-12.2f | %-10.2f\n', ...
        cat_name, aname, min(tmp_best), mean(tmp_best), std(tmp_best));
end

fprintf('\nTotal elapsed time: %.2f s\n', toc);
fprintf('========================================================\n');

%% ==================== 5. Aggregate ====================
best_overall = min(best_fit_mat, [], 2);   % (nAlgo,1)
mean_overall = mean(best_fit_mat, 2);
std_overall  = std(best_fit_mat, 0, 2);
conv_mean    = squeeze(mean(conv_mat, 2));  % (nAlgo, N_CKPT)
conv_std     = squeeze(std(conv_mat, 0, 2));

%% ==================== 6. Per-Category Winner ====================
fprintf('\n========================================================\n');
fprintf('  Per-Category Best Algorithm\n');
fprintf('========================================================\n');

cat_winner_idx  = zeros(1, nCat);   % index into ALGO_NAMES
cat_winner_name = cell(1, nCat);
cat_winner_fit  = zeros(1, nCat);

for c = 1:nCat
    members = find(algo_cat == c);          % indices of algos in this category
    [best_in_cat, local_best] = min(best_overall(members));
    winner_global = members(local_best);
    cat_winner_idx(c)  = winner_global;
    cat_winner_name{c} = ALGO_NAMES{winner_global};
    cat_winner_fit(c)  = best_in_cat;
    fprintf('  %-20s -> Winner: %-6s  (Best = %.2f m)\n', ...
        CAT_NAMES{c}, cat_winner_name{c}, best_in_cat);
end

% Overall champion across all 5 winners
[~, champ_cat] = min(cat_winner_fit);
fprintf('\n  Overall Champion: %s (%s)  -> %.2f m\n', ...
    cat_winner_name{champ_cat}, CAT_NAMES{champ_cat}, cat_winner_fit(champ_cat));
fprintf('========================================================\n\n');

%% ==================== 7. Visualization ====================
all_fit    = best_fit_mat;
run_labels = arrayfun(@(r) sprintf('Run %d',r), 1:N_RUNS, 'UniformOutput',false);

benchmark_viz_cat(all_fit, best_overall, mean_overall, std_overall, ...
    conv_mean, conv_std, ckpt_nfe, run_labels, ALGO_NAMES, ...
    CAT_NAMES, CAT_ALGOS, CAT_COLORS, algo_cat, ...
    cat_winner_idx, cat_winner_name, cat_winner_fit, ...
    NFE_total, C_GREEN, C_GOLD, C_LG, C_LY, C_W, C_MID, cmap_gg);

fprintf('\nAll figures saved to current directory.\n');


%% ============================================================
%%  Single-Algorithm Runner
%% ============================================================
function [best_fit, best_pos, conv_curve] = run_single_algo( ...
        name, fobj, dim, Lb, Ub, NFE, N_CKPT, ckpt_nfe, DFenPei_, data_)
    ep=[]; ef=[]; K=0;
    try
        switch name
            case 'GA',   [best_fit,best_pos,~,~]=run_GA(fobj,dim,Lb,Ub,NFE,ep,ef,K);
            case 'DE',   [best_fit,best_pos,~,~]=run_DE(fobj,dim,Lb,Ub,NFE,ep,ef,K);
            case 'PSO',  [best_fit,best_pos,~,~]=run_PSO(fobj,dim,Lb,Ub,NFE,ep,ef,K);
            case 'SSA',  [best_fit,best_pos,~,~]=run_SSA(fobj,dim,Lb,Ub,NFE,ep,ef,K);
            case 'GWO',  [best_fit,best_pos,~,~]=run_GWO(fobj,dim,Lb,Ub,NFE,ep,ef,K);
            case 'FA',   [best_fit,best_pos,~,~]=run_FA(fobj,dim,Lb,Ub,NFE,ep,ef,K);
            case 'ABC',  [best_fit,best_pos,~,~]=run_ABC(fobj,dim,Lb,Ub,NFE,ep,ef,K);
            case 'TOW',  [best_fit,best_pos,~,~]=run_TOW(fobj,dim,Lb,Ub,NFE,ep,ef,K);
            case 'CA',   [best_fit,best_pos,~,~]=run_CA(fobj,dim,Lb,Ub,NFE,ep,ef,K);
            case 'PO',   [best_fit,best_pos,~,~]=run_PO(fobj,dim,Lb,Ub,NFE,ep,ef,K);
            case 'CS',   [best_fit,best_pos,~,~]=run_CS(fobj,dim,Lb,Ub,NFE,ep,ef,K);
            case 'HLO',  [best_fit,best_pos,~,~]=run_HLO(fobj,dim,Lb,Ub,NFE,ep,ef,K);
            case 'SA',   [best_fit,best_pos,~,~]=run_SA_bench(fobj,dim,Lb,Ub,NFE,DFenPei_);
            case 'HS',   [best_fit,best_pos,~,~]=run_HS_bench(fobj,dim,Lb,Ub,NFE,DFenPei_,data_);
            case 'NSGA', [best_fit,best_pos,~,~]=run_NSGA_bench(fobj,dim,Lb,Ub,NFE,DFenPei_,data_);
            otherwise,   error('Unknown algorithm: %s', name);
        end
    catch ME
        best_fit = inf;
        best_pos = Lb + round(rand(1,dim).*(Ub-Lb));
        warning('Algorithm %s failed: %s', name, ME.message);
    end
    conv_curve = repmat(best_fit, 1, N_CKPT);
end


%% ============================================================
%%  Visualization — Categorized Benchmark
%% ============================================================
function benchmark_viz_cat(all_fit, best_overall, mean_overall, std_overall, ...
        conv_mean, conv_std, ckpt_nfe, run_labels, algos, ...
        CAT_NAMES, CAT_ALGOS, CAT_COLORS, algo_cat, ...
        cat_winner_idx, cat_winner_name, cat_winner_fit, ...
        NFE_total, C_GREEN, C_GOLD, C_LG, C_LY, C_W, C_MID, cmap_gg)

nAlgo  = length(algos);
nCat   = length(CAT_NAMES);
nRuns  = size(all_fit, 2);
N_CKPT = size(conv_mean, 2);

% ── Helper: colour each bar by its category ─────────────────
function c = algo_color(a_idx, algo_cat_, CAT_COLORS_)
    c = CAT_COLORS_{algo_cat_(a_idx)};
end

%% ── Fig 1: Category Overview — 5 subplots (one per category) ──
fig1 = figure('Color',C_W,'Name','fig1_category_overview', ...
    'Position',[30 30 1700 780]);

for c = 1:nCat
    ax = subplot(1,5,c);
    members = find(algo_cat == c);
    fits_c  = best_overall(members);
    [fits_s, si] = sort(fits_c,'ascend');
    names_s = algos(members(si));
    n = length(fits_s);
    rng_f = max(fits_s)-min(fits_s); if rng_f<1, rng_f=1; end

    % Winner gets full category colour; others get lighter shade
    base_c = CAT_COLORS{c};
    lite_c = 0.45*base_c + 0.55*[1 1 1];
    bar_colors = repmat(lite_c, n, 1);
    bar_colors(1,:) = base_c;    % best algo (sorted ascend → index 1)

    bh = barh(fits_s,'FaceColor','flat','EdgeColor','none','BarWidth',0.70);
    for i=1:n, bh.CData(i,:)=bar_colors(i,:); end
    hold on;

    % Crown marker on winner
    plot(fits_s(1), 1, 'p','MarkerSize',13,'MarkerFaceColor',base_c, ...
        'MarkerEdgeColor','w','LineWidth',0.6);

    % Value labels
    for i=1:n
        text(fits_s(i)+rng_f*0.01, i, sprintf('%.0f',fits_s(i)), ...
            'FontSize',8,'Color',[0.25 0.25 0.25],'VerticalAlignment','middle');
    end

    % Winner badge
    text(fits_s(1)-rng_f*0.01, 1, sprintf('  ★ BEST'), ...
        'FontSize',7.5,'Color',base_c,'FontWeight','bold', ...
        'VerticalAlignment','middle','HorizontalAlignment','right');

    yticks(1:n); yticklabels(names_s);
    xlabel('Best Distance (m)','FontSize',9);
    title(sprintf('%s', CAT_NAMES{c}), ...
        'FontSize',11,'FontWeight','bold','Color', base_c*0.7);
    set(ax,'Color',C_W,'Box','off','TickDir','out','FontSize',9, ...
        'GridAlpha',0.12,'XColor',[0.4 0.4 0.4],'YColor',[0.4 0.4 0.4]);
    grid on;
end

sgtitle(sprintf('Algorithm Benchmark by Category  (NFE=%d, Lower = Better)', NFE_total), ...
    'FontSize',13,'FontWeight','bold','Color',[0.15 0.15 0.15]);
exportgraphics(fig1,'cat_fig1_category_overview.png','Resolution',200);
fprintf('Saved: cat_fig1_category_overview.png\n');

%% ── Fig 2: Category Winners Comparison ─────────────────────
fig2 = figure('Color',C_W,'Name','fig2_winners','Position',[60 60 1100 540]);

% Left panel: horizontal bar of 5 winners
ax2l = subplot(1,2,1);
hold on; box on; grid on;
ax2l.GridColor=[0.88 0.88 0.85]; ax2l.GridAlpha=0.5;
ax2l.XColor=[0.35 0.35 0.35]; ax2l.YColor=[0.35 0.35 0.35];

[fits_w, sw] = sort(cat_winner_fit,'ascend');
names_w = cat_winner_name(sw);
cats_w  = CAT_NAMES(sw);
n_w = length(fits_w);
rng_w = max(fits_w)-min(fits_w); if rng_w<1, rng_w=1; end

bh2 = barh(fits_w,'FaceColor','flat','EdgeColor','none','BarWidth',0.65);
for k=1:n_w
    orig_cat = find(strcmp(CAT_NAMES, cats_w{k}));
    bh2.CData(k,:) = CAT_COLORS{orig_cat};
end
plot(fits_w(1), 1, 'p','MarkerSize',14,'MarkerFaceColor',C_GREEN, ...
    'MarkerEdgeColor','w','LineWidth',0.7);

for k=1:n_w
    text(fits_w(k)+rng_w*0.008, k, ...
        sprintf('%.0f m', fits_w(k)), ...
        'FontSize',9,'Color',[0.2 0.2 0.2],'VerticalAlignment','middle');
end

% Y-tick labels: "AlgoName (Category)"
ylbls = cellfun(@(n,c) sprintf('%s  (%s)',n,c), names_w, cats_w, 'UniformOutput',false);
yticks(1:n_w); yticklabels(ylbls);
xlabel('Best Overall Distance (m)','FontSize',10,'FontWeight','bold');
title('Category Winners Ranked','FontSize',11,'FontWeight','bold','Color',[0.2 0.2 0.2]);
set(ax2l,'Color',C_W,'Box','off','TickDir','out','FontSize',9);

% Right panel: grouped bar — mean ± std for each winner
ax2r = subplot(1,2,2);
hold on; box on; grid on;
ax2r.GridColor=[0.88 0.88 0.85]; ax2r.GridAlpha=0.5;
ax2r.XColor=[0.35 0.35 0.35]; ax2r.YColor=[0.35 0.35 0.35];

bw = 0.35;
w_idx = cat_winner_idx;   % global indices of 5 winners (original category order)
for c=1:nCat
    ai = w_idx(c);
    bm_ = bar(c-bw/2, mean_overall(ai), bw, 'FaceColor',CAT_COLORS{c}, ...
        'EdgeColor','none','FaceAlpha',0.90);
    bs_ = bar(c+bw/2, std_overall(ai),  bw, 'FaceColor',CAT_COLORS{c}*0.55+[1 1 1]*0.45, ...
        'EdgeColor','none','FaceAlpha',0.90);
    % Error bar on mean bar
    errorbar(c-bw/2, mean_overall(ai), std_overall(ai), ...
        'k.','LineWidth',1.2,'CapSize',5);
    text(c-bw/2, mean_overall(ai)+max(mean_overall(w_idx))*0.02, ...
        sprintf('%.0f',mean_overall(ai)),'FontSize',8, ...
        'HorizontalAlignment','center','Color',[0.2 0.2 0.2]);
    text(c+bw/2, std_overall(ai)+max(std_overall(w_idx)+1)*0.04, ...
        sprintf('%.0f',std_overall(ai)),'FontSize',8, ...
        'HorizontalAlignment','center','Color',[0.35 0.25 0.05]);
end
xticks(1:nCat);
xticklabels(cellfun(@(c,n) sprintf('%s\n(%s)',n,c), ...
    CAT_NAMES, cat_winner_name, 'UniformOutput',false));
xtickangle(0);
ylabel('Distance (m)','FontSize',10,'FontWeight','bold');
title('Winners: Mean & Std across Runs','FontSize',11,'FontWeight','bold','Color',[0.2 0.2 0.2]);
% Manual legend patches
p1=patch(NaN,NaN,C_GREEN,'EdgeColor','none'); 
p2=patch(NaN,NaN,C_GOLD*0.55+[1 1 1]*0.45,'EdgeColor','none');
legend([p1,p2],{'Mean','Std'},'Location','northeast','Box','off','FontSize',9);
set(ax2r,'Color',C_W,'Box','off','TickDir','out','FontSize',9);

sgtitle('Per-Category Winner Summary','FontSize',13,'FontWeight','bold','Color',[0.15 0.15 0.15]);
exportgraphics(fig2,'cat_fig2_winners_comparison.png','Resolution',200);
fprintf('Saved: cat_fig2_winners_comparison.png\n');

%% ── Fig 3: Full 15-Algorithm Ranking coloured by category ──
fig3 = figure('Color',C_W,'Name','fig3_full_ranking','Position',[90 50 800 820]);
ax3 = axes('Color',C_W); hold on; box on; grid on;
ax3.GridColor=[0.88 0.88 0.85]; ax3.GridAlpha=0.45;
ax3.XColor=[0.35 0.35 0.35]; ax3.YColor=[0.35 0.35 0.35];

[bo_s, si_all] = sort(best_overall,'ascend');
names_all = algos(si_all);
cats_all  = algo_cat(si_all);
n_all = length(bo_s);
rng_all = max(bo_s)-min(bo_s); if rng_all<1, rng_all=1; end

bh3 = barh(bo_s,'FaceColor','flat','EdgeColor','none','BarWidth',0.70);
for i=1:n_all
    bh3.CData(i,:) = CAT_COLORS{cats_all(i)};
end
% Winner star
plot(bo_s(1),1,'p','MarkerSize',14,'MarkerFaceColor',CAT_COLORS{cats_all(1)}, ...
    'MarkerEdgeColor','w','LineWidth',0.7);

for i=1:n_all
    text(bo_s(i)+rng_all*0.005, i, sprintf('%.0f',bo_s(i)), ...
        'FontSize',8.5,'Color',[0.25 0.25 0.25],'VerticalAlignment','middle');
    % Category badge on left
    text(bo_s(1)-rng_all*0.01, i, ...
        sprintf('[%s]', CAT_NAMES{cats_all(i)}(1:3)), ...
        'FontSize',7,'Color',CAT_COLORS{cats_all(i)}*0.65, ...
        'HorizontalAlignment','right','VerticalAlignment','middle');
end

yticks(1:n_all); yticklabels(names_all);
xlabel('Best Overall Distance (m)','FontSize',10,'FontWeight','bold');
title(sprintf('Full 15-Algorithm Ranking  (NFE=%d)',NFE_total), ...
    'FontSize',12,'FontWeight','bold','Color',[0.15 0.15 0.15]);

% Category legend
leg_patches = gobjects(nCat,1);
for c=1:nCat
    leg_patches(c)=patch(NaN,NaN,CAT_COLORS{c},'EdgeColor','none');
end
legend(leg_patches, CAT_NAMES,'Location','southeast','Box','off','FontSize',9);
set(ax3,'Color',C_W,'Box','off','TickDir','out','FontSize',9);

exportgraphics(fig3,'cat_fig3_full_ranking.png','Resolution',200);
fprintf('Saved: cat_fig3_full_ranking.png\n');

%% ── Fig 4: SHAP Heatmap (15 algos × 5 runs) ────────────────
shap_mat = zeros(nAlgo, nRuns);
for r=1:nRuns
    mu=mean(all_fit(:,r));
    shap_mat(:,r)=(mu-all_fit(:,r))/mu;
end
[~,row_order]=sort(mean(abs(shap_mat),2),'descend');
hmap=shap_mat(row_order,:);

fig4=figure('Color',C_W,'Name','fig4_shap_heatmap','Position',[120 40 980 620]);
ax4=axes('Position',[0.14 0.10 0.68 0.72],'Color',C_W);
imagesc(hmap);
colormap(ax4,cmap_gg);
clim_v=max(abs(hmap(:))); if clim_v<1e-10, clim_v=1; end
try, clim([-clim_v clim_v]); catch, caxis([-clim_v clim_v]); end
cb4=colorbar('eastoutside','Position',[0.85 0.10 0.022 0.72]);
cb4.Label.String='SHAP Value'; cb4.FontSize=8;
cb4.Ticks=[-clim_v,0,clim_v];
cb4.TickLabels={sprintf('-%.3f',clim_v),'0',sprintf('+%.3f',clim_v)};

for i=1:nAlgo
    for j=1:nRuns
        text(j,i,sprintf('%.3f',hmap(i,j)),...
            'HorizontalAlignment','center','VerticalAlignment','middle',...
            'FontSize',8,'Color',[0.12 0.12 0.12],'FontWeight','bold');
    end
end

% Colour y-tick labels by category
xticks(1:nRuns); xticklabels(run_labels);
yticks(1:nAlgo);
row_algos=algos(row_order); row_cats=algo_cat(row_order);
yticklabels(row_algos);
% Colour each ytick by category
for i=1:nAlgo
    ax4.YTickLabel{i}=row_algos{i};
end

xlabel('Independent Run','FontSize',10);
title('SHAP Heatmap — Algorithm × Run  (sorted by |SHAP|)', ...
    'FontSize',12,'FontWeight','bold','Color',[0.2 0.2 0.2]);

% Category colour strip on left
ax4s=axes('Position',[0.04 0.10 0.025 0.72],'Color','none');
for i=1:nAlgo
    patch(ax4s,[0 1 1 0],[i-0.5 i-0.5 i+0.5 i+0.5], ...
        CAT_COLORS{row_cats(i)},'EdgeColor','none');
end
xlim(ax4s,[0 1]); ylim(ax4s,[0.5 nAlgo+0.5]);
axis(ax4s,'off');
set(ax4,'TickDir','out','FontSize',9,'Box','off');

% Legend for colour strip
leg4=gobjects(nCat,1);
for c=1:nCat, leg4(c)=patch(NaN,NaN,CAT_COLORS{c},'EdgeColor','none'); end
legend(ax4,leg4,CAT_NAMES,'Location','northeast','Box','off','FontSize',8);

exportgraphics(fig4,'cat_fig4_shap_heatmap.png','Resolution',200);
fprintf('Saved: cat_fig4_shap_heatmap.png\n');

%% ── Fig 5: Per-Category SHAP Beeswarm (1 subplot per cat) ──
fig5=figure('Color',C_W,'Name','fig5_cat_beeswarm','Position',[60 60 1700 500]);
mean_shap=mean(abs(shap_mat),2);

for c=1:nCat
    ax=subplot(1,5,c);
    members=find(algo_cat==c);
    n_m=length(members);
    ms=mean_shap(members); sm=shap_mat(members,:);
    [ms_s,si_]=sort(ms,'ascend');
    sm_s=sm(si_,:); names_c=algos(members(si_));
    base_c=CAT_COLORS{c};

    hold on;
    for i=1:n_m
        barh(i,ms_s(i),0.40,'FaceColor',base_c*0.4+[1 1 1]*0.6, ...
            'EdgeColor','none','FaceAlpha',0.55);
        for j=1:nRuns
            yj=i+(rand-0.5)*0.32;
            v=max(-0.12,min(0.12,sm_s(i,j)));
            t=(v+0.12)/0.24;
            dot_c=(1-t)*C_GREEN+t*C_GOLD;
            scatter(abs(sm_s(i,j)),yj,50,dot_c,'filled', ...
                'MarkerFaceAlpha',0.85,'MarkerEdgeColor','none');
        end
        pct=ms_s(i)/max(sum(ms_s),1e-10)*100;
        text(ms_s(i)+max(ms_s)*0.02,i,sprintf('%.1f%%',pct), ...
            'FontSize',8,'Color',[0.3 0.3 0.3],'VerticalAlignment','middle');
    end
    yticks(1:n_m); yticklabels(names_c);
    xlabel('Mean |SHAP|','FontSize',9);
    title(CAT_NAMES{c},'FontSize',11,'FontWeight','bold','Color',base_c*0.7);
    set(ax,'Color',C_W,'Box','off','TickDir','out','FontSize',9, ...
        'GridAlpha',0.12,'XColor',[0.4 0.4 0.4],'YColor',[0.4 0.4 0.4]);
    grid on; ax.XGrid='on'; ax.YGrid='off';
end
sgtitle('Per-Category SHAP Contribution (Beeswarm)', ...
    'FontSize',13,'FontWeight','bold','Color',[0.15 0.15 0.15]);
exportgraphics(fig5,'cat_fig5_cat_beeswarm.png','Resolution',200);
fprintf('Saved: cat_fig5_cat_beeswarm.png\n');

%% ── Fig 6: Pairwise Interaction Matrix (15×15, cat-coloured) ─
cross_mat=zeros(nAlgo,nAlgo);
for i=1:nAlgo
    for j=1:nAlgo
        if i~=j, cross_mat(i,j)=mean(all_fit(j,:)-all_fit(i,:)); end
    end
end
fig6=figure('Color',C_W,'Name','fig6_interaction','Position',[150 30 1050 920]);
ax6=axes('Color',C_W);
imagesc(cross_mat);
colormap(ax6,cmap_gg);
mx6=max(abs(cross_mat(:))); if mx6<1, mx6=1; end
try, clim([-mx6 mx6]); catch, caxis([-mx6 mx6]); end
cb6=colorbar;
cb6.Label.String='Performance Gap (m,  + = row beats column)'; cb6.FontSize=8;
for i=1:nAlgo
    for j=1:nAlgo
        v=cross_mat(i,j);
        if i~=j && abs(v)>mx6*0.15
            text(j,i,sprintf('%.0f',v),'HorizontalAlignment','center', ...
                'VerticalAlignment','middle','FontSize',7,'Color',[0.12 0.12 0.12]);
        end
    end
end
xticks(1:nAlgo); xticklabels(algos); xtickangle(38);
yticks(1:nAlgo); yticklabels(algos);
xlabel('Reference Algorithm (Column)','FontSize',10);
ylabel('Target Algorithm (Row)','FontSize',10);
title('Algorithm Pairwise Interaction Matrix','FontSize',12,'FontWeight','bold','Color',[0.2 0.2 0.2]);
set(ax6,'TickDir','out','FontSize',9,'Box','off');
exportgraphics(fig6,'cat_fig6_pairwise_matrix.png','Resolution',200);
fprintf('Saved: cat_fig6_pairwise_matrix.png\n');

%% ── Fig 7: Winners Radar + Convergence ──────────────────────
fig7=figure('Color',C_W,'Name','fig7_winners_radar','Position',[200 80 1400 680]);

% Left: Radar — 5 category winners, axes = independent runs
ax7l=subplot(1,2,1);
axes('Color',C_W,'Position',[0.04 0.08 0.44 0.84]); hold on;
theta=linspace(0,2*pi,nRuns+1); theta(end)=[];
for rv=[0.25,0.5,0.75,1.0]
    th_=linspace(0,2*pi,200);
    plot(cos(th_)*rv,sin(th_)*rv,'-','Color',[0.88 0.88 0.85],'LineWidth',0.7);
end
for r=1:nRuns
    [xe,ye]=pol2cart(theta(r),1.0);
    plot([0 xe],[0 ye],'--','Color',[0.82 0.82 0.80],'LineWidth',0.8);
    [xl_,yl_]=pol2cart(theta(r),1.24);
    text(xl_,yl_,run_labels{r},'HorizontalAlignment','center', ...
        'FontSize',9,'Color',[0.28 0.28 0.28],'FontWeight','bold');
end

% Score matrix for winners only
score_w=zeros(nCat,nRuns);
for r=1:nRuns
    vals=all_fit(cat_winner_idx,r);
    [~,si_]=sort(vals,'ascend');
    rnk=zeros(nCat,1); rnk(si_)=(nCat:-1:1)';
    score_w(:,r)=(rnk-1)/max(nCat-1,1);
end

for c=1:nCat
    vals=score_w(c,:);
    valsc=[vals,vals(1)]; thetac=[theta,theta(1)];
    [xp,yp]=pol2cart(thetac,valsc);
    fill(xp,yp,CAT_COLORS{c},'FaceAlpha',0.12,'EdgeColor','none');
    plot(xp,yp,'-o','Color',CAT_COLORS{c},'LineWidth',2.5,'MarkerSize',7, ...
        'MarkerFaceColor',CAT_COLORS{c},'MarkerEdgeColor','w');
end
leg7=gobjects(nCat,1);
for c=1:nCat
    leg7(c)=plot(NaN,NaN,'-o','Color',CAT_COLORS{c},'LineWidth',2.5, ...
        'MarkerFaceColor',CAT_COLORS{c},'MarkerEdgeColor','w','MarkerSize',7);
end
leg_labels=cellfun(@(c,n) sprintf('%s (%s)',n,c), ...
    CAT_NAMES,cat_winner_name,'UniformOutput',false);
legend(leg7,leg_labels,'Location','southoutside','Orientation','horizontal', ...
    'FontSize',8.5,'Box','off');
axis equal off;
title('Category Winners — Run Score Radar','FontSize',11,'FontWeight','bold','Color',[0.2 0.2 0.2]);

% Right: Convergence curves for 5 winners
ax7r=subplot(1,2,2);
set(ax7r,'Color',C_W,'Box','off','TickDir','out','FontSize',9, ...
    'GridAlpha',0.12,'XColor',[0.35 0.35 0.35],'YColor',[0.35 0.35 0.35]);
hold(ax7r,'on'); grid(ax7r,'on');

leg_h=gobjects(nCat,1);
for c=1:nCat
    ai=cat_winner_idx(c);
    ym=conv_mean(ai,:); ys=conv_std(ai,:); xv=ckpt_nfe;
    fill(ax7r,[xv,fliplr(xv)],[ym+ys,fliplr(ym-ys)], ...
        CAT_COLORS{c},'FaceAlpha',0.13,'EdgeColor','none');
    leg_h(c)=plot(ax7r,xv,ym,'-o','Color',CAT_COLORS{c},'LineWidth',2.3, ...
        'MarkerSize',4,'MarkerFaceColor',CAT_COLORS{c},'MarkerEdgeColor','w');
end
xlabel(ax7r,'NFE','FontSize',10,'FontWeight','bold');
ylabel(ax7r,'Best Fitness (m)','FontSize',10,'FontWeight','bold');
title(ax7r,'Convergence Curves — Category Winners (Mean ± Std)', ...
    'FontSize',11,'FontWeight','bold','Color',[0.2 0.2 0.2]);
legend(ax7r,leg_h,leg_labels,'Location','northeast','Box','off','FontSize',9);

exportgraphics(fig7,'cat_fig7_winners_radar_convergence.png','Resolution',200);
fprintf('Saved: cat_fig7_winners_radar_convergence.png\n');

end  % benchmark_viz_cat


%% ============================================================
%%  15 Algorithm Implementations (same as parallel_benchmark.m)
%% ============================================================

function pop=init_pop_bench(popSize,dim,Lb,Ub)
    pop = Lb + round(rand(popSize,dim).*(Ub-Lb));
end

function [best_fit,best_pos,pop,fit]=run_GA(fobj,dim,Lb,Ub,NFE,~,~,~)
    popSize=60; maxGen=floor(NFE/popSize); pc=0.85; pm=0.12;
    pop=init_pop_bench(popSize,dim,Lb,Ub);
    fit=arrayfun(@(i) fobj(pop(i,:)),1:popSize)';
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);
    for iter_=1:maxGen
        newP=zeros(size(pop));
        for i=1:popSize
            c=randperm(popSize,3); [~,w]=min(fit(c)); newP(i,:)=pop(c(w),:);
        end
        for i=1:2:popSize-1
            if rand<pc
                cp=randi(dim);
                tmp=newP(i,cp:end); newP(i,cp:end)=newP(i+1,cp:end); newP(i+1,cp:end)=tmp;
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

function [best_fit,best_pos,pop,fit]=run_DE(fobj,dim,Lb,Ub,NFE,~,~,~)
    popSize=50; maxGen=floor(NFE/popSize); F=0.7; CR=0.8;
    pop=init_pop_bench(popSize,dim,Lb,Ub);
    fit=arrayfun(@(i) fobj(pop(i,:)),1:popSize)';
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);
    for iter_=1:maxGen
        for i=1:popSize
            cands=setdiff(1:popSize,i);
            r=cands(randperm(length(cands),3));
            v=max(Lb,min(Ub,round(pop(r(1),:)+F*(pop(r(2),:)-pop(r(3),:)))));
            mask=rand(1,dim)<CR; mask(randi(dim))=true;
            trial=pop(i,:); trial(mask)=v(mask);
            trial=max(Lb,min(Ub,trial));
            ft=fobj(trial);
            if ft<fit(i), pop(i,:)=trial; fit(i)=ft; end
            if ft<best_fit, best_fit=ft; best_pos=trial; end
        end
    end
end

function [best_fit,best_pos,pop,fit]=run_PSO(fobj,dim,Lb,Ub,NFE,~,~,~)
    popSize=50; maxIter=floor(NFE/popSize); w=0.8; c1=1.5; c2=1.5;
    pop=init_pop_bench(popSize,dim,Lb,Ub);
    vel=zeros(popSize,dim);
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

function [best_fit,best_pos,pop,fit]=run_SSA(fobj,dim,Lb,Ub,NFE,~,~,~)
    popSize=50; maxIter=floor(NFE/popSize); P_pct=0.2;
    pop=double(init_pop_bench(popSize,dim,Lb,Ub));
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

function [best_fit,best_pos,pop,fit]=run_GWO(fobj,dim,Lb,Ub,NFE,~,~,~)
    n=50; maxIter=floor(NFE/n);
    pop=init_pop_bench(n,dim,Lb,Ub);
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
            if f<As, Ds=Bs;Dp=Bp;Bs=As;Bp=Ap;As=f;Ap=np;
            elseif f<Bs, Ds=Bs;Dp=Bp;Bs=f;Bp=np;
            elseif f<Ds, Ds=f;Dp=np; end
        end
        if As<best_fit, best_fit=As; best_pos=Ap; end
    end
end

function [best_fit,best_pos,pop,fit]=run_FA(fobj,dim,Lb,Ub,NFE,~,~,~)
    n=40; maxIter=floor(NFE/n); alpha=0.5; betamin=0.2; gamma=1;
    pop=double(init_pop_bench(n,dim,Lb,Ub));
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
                    pop(i,:)=max(Lb,min(Ub,pop(i,:)));
                    fit(i)=fobj(pop(i,:));
                end
            end
        end
        [fmin,bi]=min(fit);
        if fmin<best_fit, best_fit=fmin; best_pos=pop(bi,:); end
    end
end

function [best_fit,best_pos,pop,fit]=run_ABC(fobj,dim,Lb,Ub,NFE,~,~,~)
    FN=25; limit=20; maxCycle=floor(NFE/(FN*2));
    pop=init_pop_bench(FN,dim,Lb,Ub);
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
                pop(i,:)=Lb+round(rand(1,dim).*(Ub-Lb));
                fit(i)=fobj(pop(i,:)); trial(i)=0;
                if fit(i)<best_fit, best_fit=fit(i); best_pos=pop(i,:); end
            end
        end
    end
end

function [best_fit,best_pos,pop,fit]=run_TOW(fobj,dim,Lb,Ub,NFE,~,~,~)
    nT=30; maxIter=floor(NFE/nT); alpha_t=0.98; sigma0=2.0;
    pop=double(init_pop_bench(nT,dim,Lb,Ub));
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

function [best_fit,best_pos,pop,fit]=run_CA(fobj,dim,Lb,Ub,NFE,~,~,~)
    nPop=50; nAccept=round(0.2*nPop); alpha_ca=0.15;
    maxIter=floor(NFE/nPop);
    pop=double(init_pop_bench(nPop,dim,Lb,Ub));
    fit=arrayfun(@(i) fobj(pop(i,:)),1:nPop)';
    [~,si]=sort(fit); best_fit=fit(si(1)); best_pos=pop(si(1),:);
    cult_best=best_pos; norm_lo=Lb; norm_hi=Ub;
    for iter_=1:maxIter
        for i=1:nPop
            sigma=alpha_ca*(norm_hi-norm_lo);
            dx=round(sigma.*randn(1,dim))+round(0.3*sign(cult_best-pop(i,:)));
            pop(i,:)=max(Lb,min(Ub,round(pop(i,:)+dx)));
            fit(i)=fobj(pop(i,:));
        end
        [~,si]=sort(fit);
        spop=pop(si(1:nAccept),:);
        norm_lo=min(spop,[],1); norm_hi=max(spop,[],1)+1;
        if fit(si(1))<best_fit
            best_fit=fit(si(1)); best_pos=pop(si(1),:); cult_best=best_pos;
        end
    end
end

function [best_fit,best_pos,pop,fit]=run_PO(fobj,dim,Lb,Ub,NFE,~,~,~)
    parties=10; areas=10; popSize=parties*areas;
    maxIter=floor(NFE/popSize);
    pop=init_pop_bench(popSize,dim,Lb,Ub);
    fit=arrayfun(@(i) fobj(pop(i,:)),1:popSize)';
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);
    for it=1:maxIter
        for p=1:parties
            idx_p=(p-1)*areas+1:p*areas;
            [~,ll]=min(fit(idx_p)); leader_idx=idx_p(ll); leader=pop(leader_idx,:);
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
            idx_p=(p-1)*areas+1:p*areas;
            [~,ll]=min(fit(idx_p)); li=idx_p(ll);
            step=round((global_leader-pop(li,:))*0.25*rand+randn(1,dim)*0.6);
            np=max(Lb,min(Ub,pop(li,:)+step)); f=fobj(np);
            if f<fit(li), pop(li,:)=np; fit(li)=f; end
            if f<best_fit, best_fit=f; best_pos=np; end
        end
    end
end

function [best_fit,best_pos,pop,fit]=run_CS(fobj,dim,Lb,Ub,NFE,~,~,~)
    n=30; maxIter=floor(NFE/n); pa=0.25; beta=1.5;
    pop=init_pop_bench(n,dim,Lb,Ub);
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
                np=Lb+round(rand(1,dim).*(Ub-Lb));
                fit(i)=fobj(np); pop(i,:)=np;
                if fit(i)<best_fit, best_fit=fit(i); best_pos=np; end
            end
        end
    end
end

function levy=levy_cs(beta,dim)
    sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u=randn(1,dim)*sigma; v=randn(1,dim);
    levy=u./(abs(v).^(1/beta));
    levy=sign(levy).*min(abs(levy),5);
end

function [best_fit,best_pos,pop,fit]=run_HLO(fobj,dim,Lb,Ub,NFE,~,~,~)
    popSize=40; bpv=8; m=dim*bpv;
    maxIter=floor(NFE/popSize); p_r=0.1; p_i=0.5;
    bin_pop=randi([0,1],popSize,m);
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

function [best_fit,best_pos,pop,fit]=run_SA_bench(fobj,dim,Lb,Ub,NFE,DFenPei_)
    T0=1e5; Tend=1; q=0.93; L=30; NFE_used=0;
    all_cands=unique(cell2mat(cellfun(@(c) c(2:end),DFenPei_,'UniformOutput',false)));
    nCands=length(all_cands); num_sel=min(57,nCands);
    S1=all_cands(randperm(nCands,num_sel));
    x_int1=sa_to_x(S1,DFenPei_,Lb,Ub);
    best_fit=fobj(x_int1); NFE_used=1; best_pos=x_int1;
    while T0>Tend && NFE_used<NFE
        for i=1:L
            S2=S1; unsel=setdiff(all_cands,S1);
            if isempty(unsel), break; end
            S2(randi(num_sel))=unsel(randi(length(unsel)));
            x2=sa_to_x(S2,DFenPei_,Lb,Ub);
            f2=fobj(x2); NFE_used=NFE_used+1;
            delta=f2-best_fit;
            if f2<best_fit||exp(-delta/T0)>rand
                S1=S2;
                if f2<best_fit, best_fit=f2; best_pos=x2; end
            end
            if NFE_used>=NFE, break; end
        end
        T0=T0*q;
    end
    pop_size=min(20,dim);
    pop=repmat(best_pos,pop_size,1);
    for i=2:pop_size
        pop(i,:)=max(Lb,min(Ub,best_pos+round(randn(1,dim))));
    end
    fit=arrayfun(@(i) fobj(pop(i,:)),1:pop_size)';
end

function x_int=sa_to_x(sel,DFenPei_,Lb,Ub)
    x_int=zeros(1,length(DFenPei_));
    for i=1:length(DFenPei_)
        cands=DFenPei_{i}(2:end);
        overlap=intersect(cands,sel);
        if ~isempty(overlap), idx=find(DFenPei_{i}==overlap(1))-1;
        else, idx=1; end
        x_int(i)=max(Lb(i),min(Ub(i),idx));
    end
end

function [best_fit,best_pos,pop,fit]=run_HS_bench(fobj,dim,Lb,Ub,NFE,DFenPei_,data_)
    try
        binan_xy=data_.binan; start_xy=data_.start;
    catch
        pop=init_pop_bench(20,dim,Lb,Ub);
        fit=arrayfun(@(i) fobj(pop(i,:)),1:20)';
        [best_fit,bi]=min(fit); best_pos=pop(bi,:); return;
    end
    HMS=15; num_centers=min(57,size(binan_xy,1));
    house_x=start_xy(:,1); house_y=start_xy(:,2);
    min_x=min(house_x); max_x=max(house_x);
    min_y=min(house_y); max_y=max(house_y);
    NVAR=num_centers*2; NFE_used=0;
    BW_max=(max_x-min_x)*0.2; BW_min=(max_x-min_x)*0.0001;
    maxItr=floor(NFE/HMS);
    HM=zeros(HMS,NVAR); hm_fit=zeros(HMS,1);
    for i=1:HMS
        idx_r=randperm(size(house_x,1),num_centers);
        pos=[house_x(idx_r),house_y(idx_r)]; HM(i,:)=pos(:)';
        x_=hs_to_int(HM(i,:),num_centers,DFenPei_,binan_xy,Lb,Ub);
        hm_fit(i)=fobj(x_); NFE_used=NFE_used+1;
    end
    [best_hf,bi]=min(hm_fit); best_harm=HM(bi,:);
    for itr=1:maxItr
        if NFE_used>=NFE, break; end
        BW=BW_max*exp(log(BW_min/BW_max)*itr/maxItr);
        [~,bi_]=min(hm_fit); new_h=HM(bi_,:);
        t=randi(num_centers); ix=t*2-1; iy=t*2;
        new_h(ix)=new_h(ix)+(rand*2-1)*BW; new_h(iy)=new_h(iy)+(rand*2-1)*BW;
        new_h(ix)=max(min(new_h(ix),max_x),min_x);
        new_h(iy)=max(min(new_h(iy),max_y),min_y);
        x_=hs_to_int(new_h,num_centers,DFenPei_,binan_xy,Lb,Ub);
        fnew=fobj(x_); NFE_used=NFE_used+1;
        [worst_f,wi]=max(hm_fit);
        if fnew<worst_f
            HM(wi,:)=new_h; hm_fit(wi)=fnew;
            if fnew<best_hf, best_hf=fnew; best_harm=new_h; end
        end
    end
    best_pos=hs_to_int(best_harm,num_centers,DFenPei_,binan_xy,Lb,Ub);
    best_fit=fobj(best_pos);
    pop_size=min(20,HMS); pop=repmat(best_pos,pop_size,1);
    for i=2:pop_size
        pop(i,:)=max(Lb,min(Ub,best_pos+round(randn(1,dim))));
    end
    fit=arrayfun(@(i) fobj(pop(i,:)),1:pop_size)';
end

function x_int=hs_to_int(harm,nc,DFenPei_,binan_xy,Lb,Ub)
    centers=reshape(harm,[],2); sel=zeros(1,nc);
    for c=1:nc
        d=sqrt((binan_xy(:,1)-centers(c,1)).^2+(binan_xy(:,2)-centers(c,2)).^2);
        [~,mi]=min(d); sel(c)=mi;
    end
    sel=unique(sel);
    x_int=zeros(1,length(DFenPei_));
    for i=1:length(DFenPei_)
        cands=DFenPei_{i}(2:end);
        overlap=intersect(cands,sel);
        if ~isempty(overlap), idx=find(DFenPei_{i}==overlap(1))-1;
        else, idx=1; end
        x_int(i)=max(Lb(i),min(Ub(i),idx));
    end
end

function [best_fit,best_pos,pop,fit]=run_NSGA_bench(fobj,dim,Lb,Ub,NFE,DFenPei_,data_)
    try, dis_mat_=data_.dis;
    catch
        pop=init_pop_bench(20,dim,Lb,Ub);
        fit=arrayfun(@(i) fobj(pop(i,:)),1:20)';
        [best_fit,bi]=min(fit); best_pos=pop(bi,:); return;
    end
    popSize=min(100,floor(NFE/20)); maxGen=floor(NFE/popSize);
    P_=cellfun(@(x) length(x)-1,DFenPei_);
    nsga_obj=@(x) nsga_eval(x,DFenPei_,dis_mat_,P_);
    pop=Lb+round(rand(popSize,dim).*(Ub-Lb));
    F=zeros(popSize,2);
    for i=1:popSize, F(i,:)=nsga_obj(pop(i,:)); end
    for g=1:maxGen
        child=zeros(popSize,dim);
        for i=1:popSize
            p1=pop(randi(popSize),:); p2=pop(randi(popSize),:);
            cp=randi(dim); child(i,:)=[p1(1:cp-1),p2(cp:end)];
            if rand<0.1
                d_=randi(dim);
                child(i,d_)=Lb(d_)+randi(max(1,Ub(d_)-Lb(d_)+1))-1;
            end
            child(i,:)=max(Lb,min(Ub,child(i,:)));
        end
        combined=[pop;child]; Fc=zeros(2*popSize,2);
        for i=1:2*popSize, Fc(i,:)=nsga_obj(combined(i,:)); end
        [~,si]=sort(Fc(:,1));
        pop=combined(si(1:popSize),:); F=Fc(si(1:popSize),:);
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

function fitness=unified_fobj(x,DFenPei,dis_mat,Lb,Ub)
    fitness=0;
    for i=1:length(DFenPei)
        idx=max(Lb(i),min(round(x(i)),Ub(i)));
        fitness=fitness+dis_mat(DFenPei{i}(1),DFenPei{i}(idx+1));
    end
end