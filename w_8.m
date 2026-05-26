%% ============================================================
%  Pipeline Integrative Hybrid — ABC → NSGA × 4
%
%  Optimized five-stage combination:
%    Stage 1 (Global Exploration)          → ABC
%    Stage 2 (Hierarchical Convergence)    → NSGA
%    Stage 3 (Stagnation Breakout)         → NSGA
%    Stage 4 (Neighborhood Intensification)→ NSGA
%    Stage 5 (Deep Convergence)            → NSGA
%
%  Elite pool transfer: Top-K elites passed between stages.
%  Objective: integer-encoded total evacuation distance.
%  Output: 8 evaluation metrics + convergence curve +
%          stage contribution chart + 2D allocation map.
%% ============================================================
clc; clear; close all; tic;

fprintf('========================================================\n');
fprintf('  Pipeline Integrative Hybrid\n');
fprintf('  Stage1:ABC -> Stage2:NSGA -> Stage3:NSGA\n');
fprintf('          -> Stage4:NSGA -> Stage5:NSGA\n');
fprintf('========================================================\n\n');

%% ======================== 0. Data Loading ========================
if exist('sj5.mat','file'), load('sj5.mat');
else, error('sj5.mat not found'); end

if exist('dis','var'), data.dis = dis; end

raw_x = data.start(:,1); raw_y = data.start(:,2);
offset_x = min(raw_x);   offset_y = min(raw_y);
house_x  = raw_x - offset_x;   house_y = raw_y - offset_y;
binan_x  = data.binan(:,1) - offset_x;
binan_y  = data.binan(:,2) - offset_y;

dim = length(DFenPei);
Lb  = ones(1, dim);
Ub  = arrayfun(@(i) length(DFenPei{i})-1, 1:dim);

% Fixed assignments
FID=[]; alldis_fixed=0; YFP=zeros(1,size(data.binan,1));
for k=1:length(B)
    if length(B{k})==1
        tb=B{k}; FID=[FID;k,tb];
        alldis_fixed=alldis_fixed+data.dis(k,tb);
        YFP(tb)=YFP(tb)+12;
    end
end

fobj = @(x) unified_fobj(x, DFenPei, data.dis, Lb, Ub);

%% ======================== Parameters ========================
K_elite = 30;

% NFE budget (front-loaded exploration, rear-loaded convergence)
budget = struct();
budget.S1_ABC   = 20000;   % Stage 1 — global diversity
budget.S2_NSGA  = 25000;   % Stage 2 — multi-obj guided contraction
budget.S3_NSGA  = 25000;   % Stage 3 — stagnation breakout
budget.S4_NSGA  = 30000;   % Stage 4 — neighborhood intensification
budget.S5_NSGA  = 50000;   % Stage 5 — deep convergence (largest budget)

stage_labels = {'Global Exploration',          ...
                'Hierarchical Convergence',     ...
                'Stagnation Breakout',          ...
                'Neighborhood Intensification', ...
                'Deep Convergence'};
stage_names  = {'ABC','NSGA','NSGA','NSGA','NSGA'};

% ── Unified green-gold palette (matches pipeline_viz) ──────────
%    S1 dark-green  S2 mid-green  S3 light-green  S4 light-gold  S5 dark-gold
C_GREEN  = [88,  140,  90] /255;
C_GOLD   = [214, 164,  59] /255;
C_MID    = 0.5*C_GREEN + 0.5*C_GOLD;   % pre-compute S2 blend (avoids vertcat)
C_LG     = [160, 200, 130] /255;
C_LY     = [240, 210, 140] /255;
stage_colors = {C_GREEN, C_MID, C_LG, C_LY, C_GOLD};

CC_all = []; stage_boundary = zeros(1,5);
P_     = cellfun(@(x) length(x)-1, DFenPei);   % upper bound per gene

%% ============================================================
%  STAGE 1: ABC — Global Exploration
%  Random initialization, artificial bee colony diversity.
%% ============================================================
fprintf('[Stage 1] ABC  — Global Exploration...\n');

FN1    = 40;
limit1 = 25;
maxCyc1= floor(budget.S1_ABC / (FN1*2));

pop1 = zeros(FN1, dim);
for i=1:FN1
    pop1(i,:) = Lb + round(rand(1,dim).*(Ub-Lb));
end
fit1   = arrayfun(@(i) fobj(pop1(i,:)), 1:FN1)';
trial1 = zeros(1,FN1);
[bestFit1, bi] = min(fit1);
bestPos1 = pop1(bi,:);
CC1 = zeros(maxCyc1,1);

for iter=1:maxCyc1
    % Employed bees
    for i=1:FN1
        k=i; while k==i, k=randi(FN1); end
        phi = randi([-1,1],1,dim);
        np  = max(Lb, min(Ub, round(pop1(i,:) + phi.*(pop1(i,:)-pop1(k,:)))));
        fn  = fobj(np);
        if fn < fit1(i)
            pop1(i,:)=np; fit1(i)=fn; trial1(i)=0;
        else
            trial1(i)=trial1(i)+1;
        end
        if fn < bestFit1, bestFit1=fn; bestPos1=np; end
    end
    % Scout bees (abandon & random re-init)
    for i=1:FN1
        if trial1(i) > limit1
            pop1(i,:) = Lb + round(rand(1,dim).*(Ub-Lb));
            fit1(i)   = fobj(pop1(i,:));
            trial1(i) = 0;
            if fit1(i) < bestFit1, bestFit1=fit1(i); bestPos1=pop1(i,:); end
        end
    end
    CC1(iter) = bestFit1;
end

[~,si]=sort(fit1);
elite_pop = pop1(si(1:min(K_elite,FN1)),:);
elite_fit = fit1(si(1:min(K_elite,FN1)));

CC_all=[CC_all;CC1]; stage_boundary(1)=length(CC_all);
fprintf('  ABC done  | Best: %.2f m | Elite pool: %d solutions\n\n', ...
    bestFit1, size(elite_pop,1));

%% ============================================================
%  STAGE 2: NSGA — Hierarchical Convergence
%  Receives Stage 1 elites. Pareto front guides population
%  toward promising regions. Moderate mutation rate.
%% ============================================================
fprintf('[Stage 2] NSGA — Hierarchical Convergence...\n');

popSz2  = 60;
maxGen2 = floor(budget.S2_NSGA / popSz2);
mut_r2  = 0.08;   % moderate mutation

pop2 = elite_inject(popSz2, dim, Lb, Ub, elite_pop, elite_fit, K_elite);
fit2 = arrayfun(@(i) fobj(pop2(i,:)), 1:popSz2)';
[bestFit2, bi] = min(fit2); bestPos2=pop2(bi,:);
CC2 = zeros(maxGen2,1);

for g=1:maxGen2
    child2 = zeros(popSz2,dim);
    for i=1:popSz2
        p1=pop2(randi(popSz2),:); p2=pop2(randi(popSz2),:);
        cp=randi(dim);
        child2(i,:)=[p1(1:cp-1), p2(cp:end)];
        if rand<mut_r2
            j=randi(dim);
            child2(i,j)=Lb(j)+randi(Ub(j)-Lb(j)+1)-1;
        end
        child2(i,:)=max(Lb,min(Ub,child2(i,:)));
    end
    combined=[pop2;child2];
    Fc2=zeros(2*popSz2,2);
    for i=1:2*popSz2
        Fc2(i,:)=nsga_dual_obj(combined(i,:), DFenPei, data.dis, P_);
    end
    [~,si]=sort(Fc2(:,1));
    pop2=combined(si(1:popSz2),:);
    fit2=arrayfun(@(i) fobj(pop2(i,:)), 1:popSz2)';
    [cf,bi]=min(fit2);
    if cf<bestFit2, bestFit2=cf; bestPos2=pop2(bi,:); end
    CC2(g)=bestFit2;
end

[~,si]=sort(fit2);
elite_pop=pop2(si(1:min(K_elite,popSz2)),:);
elite_fit=fit2(si(1:min(K_elite,popSz2)));

CC_all=[CC_all;CC2]; stage_boundary(2)=length(CC_all);
fprintf('  NSGA done | Best: %.2f m\n\n', bestFit2);

%% ============================================================
%  STAGE 3: NSGA — Stagnation Breakout
%  Higher mutation rate to escape Stage 2 local optima.
%  Stronger perturbation while retaining elite structure.
%% ============================================================
fprintf('[Stage 3] NSGA — Stagnation Breakout...\n');

popSz3  = 60;
maxGen3 = floor(budget.S3_NSGA / popSz3);
mut_r3  = 0.15;   % elevated mutation for breakout

pop3 = elite_inject(popSz3, dim, Lb, Ub, elite_pop, elite_fit, K_elite);
fit3 = arrayfun(@(i) fobj(pop3(i,:)), 1:popSz3)';
[bestFit3, bi] = min(fit3); bestPos3=pop3(bi,:);
CC3 = zeros(maxGen3,1);

for g=1:maxGen3
    child3 = zeros(popSz3,dim);
    for i=1:popSz3
        p1=pop3(randi(popSz3),:); p2=pop3(randi(popSz3),:);
        cp=randi(dim);
        child3(i,:)=[p1(1:cp-1), p2(cp:end)];
        % Per-gene mutation for stronger diversity
        for j=1:dim
            if rand<mut_r3
                child3(i,j)=Lb(j)+randi(Ub(j)-Lb(j)+1)-1;
            end
        end
        child3(i,:)=max(Lb,min(Ub,child3(i,:)));
    end
    combined=[pop3;child3];
    Fc3=zeros(2*popSz3,2);
    for i=1:2*popSz3
        Fc3(i,:)=nsga_dual_obj(combined(i,:), DFenPei, data.dis, P_);
    end
    [~,si]=sort(Fc3(:,1));
    pop3=combined(si(1:popSz3),:);
    fit3=arrayfun(@(i) fobj(pop3(i,:)), 1:popSz3)';
    [cf,bi]=min(fit3);
    if cf<bestFit3, bestFit3=cf; bestPos3=pop3(bi,:); end
    CC3(g)=bestFit3;
end

[~,si]=sort(fit3);
elite_pop=pop3(si(1:min(K_elite,popSz3)),:);
elite_fit=fit3(si(1:min(K_elite,popSz3)));

CC_all=[CC_all;CC3]; stage_boundary(3)=length(CC_all);
fprintf('  NSGA done | Best: %.2f m\n\n', bestFit3);

%% ============================================================
%  STAGE 4: NSGA — Neighborhood Intensification
%  Reduced mutation rate + larger population to compress
%  the search space around the current best neighborhood.
%% ============================================================
fprintf('[Stage 4] NSGA — Neighborhood Intensification...\n');

popSz4  = 80;   % larger pool for fine-grained search
maxGen4 = floor(budget.S4_NSGA / popSz4);
mut_r4  = 0.04;  % reduced mutation — tighter exploitation

pop4 = elite_inject(popSz4, dim, Lb, Ub, elite_pop, elite_fit, K_elite);
fit4 = arrayfun(@(i) fobj(pop4(i,:)), 1:popSz4)';
[bestFit4, bi] = min(fit4); bestPos4=pop4(bi,:);
CC4 = zeros(maxGen4,1);

for g=1:maxGen4
    child4 = zeros(popSz4,dim);
    for i=1:popSz4
        p1=pop4(randi(popSz4),:); p2=pop4(randi(popSz4),:);
        cp=randi(dim);
        child4(i,:)=[p1(1:cp-1), p2(cp:end)];
        if rand<mut_r4
            j=randi(dim);
            child4(i,j)=Lb(j)+randi(Ub(j)-Lb(j)+1)-1;
        end
        child4(i,:)=max(Lb,min(Ub,child4(i,:)));
    end
    combined=[pop4;child4];
    Fc4=zeros(2*popSz4,2);
    for i=1:2*popSz4
        Fc4(i,:)=nsga_dual_obj(combined(i,:), DFenPei, data.dis, P_);
    end
    [~,si]=sort(Fc4(:,1));
    pop4=combined(si(1:popSz4),:);
    fit4=arrayfun(@(i) fobj(pop4(i,:)), 1:popSz4)';
    [cf,bi]=min(fit4);
    if cf<bestFit4, bestFit4=cf; bestPos4=pop4(bi,:); end
    CC4(g)=bestFit4;
end

[~,si]=sort(fit4);
elite_pop=pop4(si(1:min(K_elite,popSz4)),:);
elite_fit=fit4(si(1:min(K_elite,popSz4)));

CC_all=[CC_all;CC4]; stage_boundary(4)=length(CC_all);
fprintf('  NSGA done | Best: %.2f m\n\n', bestFit4);

%% ============================================================
%  STAGE 5: NSGA — Deep Convergence
%  Largest budget. Very low mutation rate + largest population.
%  Final Pareto-guided squeeze toward global optimum.
%% ============================================================
fprintf('[Stage 5] NSGA — Deep Convergence...\n');

popSz5  = 100;   % maximum population for final sweep
maxGen5 = floor(budget.S5_NSGA / popSz5);
mut_r5  = 0.02;  % minimal mutation — pure exploitation

pop5 = elite_inject(popSz5, dim, Lb, Ub, elite_pop, elite_fit, K_elite);
fit5 = arrayfun(@(i) fobj(pop5(i,:)), 1:popSz5)';
[bestFit5, bi] = min(fit5); bestPos5=pop5(bi,:);
CC5 = zeros(maxGen5,1);

for g=1:maxGen5
    child5 = zeros(popSz5,dim);
    for i=1:popSz5
        p1=pop5(randi(popSz5),:); p2=pop5(randi(popSz5),:);
        cp=randi(dim);
        child5(i,:)=[p1(1:cp-1), p2(cp:end)];
        if rand<mut_r5
            j=randi(dim);
            child5(i,j)=Lb(j)+randi(Ub(j)-Lb(j)+1)-1;
        end
        child5(i,:)=max(Lb,min(Ub,child5(i,:)));
    end
    combined=[pop5;child5];
    Fc5=zeros(2*popSz5,2);
    for i=1:2*popSz5
        Fc5(i,:)=nsga_dual_obj(combined(i,:), DFenPei, data.dis, P_);
    end
    [~,si]=sort(Fc5(:,1));
    pop5=combined(si(1:popSz5),:);
    fit5=arrayfun(@(i) fobj(pop5(i,:)), 1:popSz5)';
    [cf,bi]=min(fit5);
    if cf<bestFit5, bestFit5=cf; bestPos5=pop5(bi,:); end
    CC5(g)=bestFit5;
end

X_final = bestPos5;
CC_all=[CC_all;CC5]; stage_boundary(5)=length(CC_all);
fprintf('  NSGA done | Best: %.2f m\n\n', bestFit5);

%% ======================== Metrics (8 KPIs) ========================
fprintf('========================================================\n');
fprintf('  Per-Stage Improvement Summary\n');
fprintf('========================================================\n');

stage_starts = [1, stage_boundary(1:4)+1];
for s=1:5
    seg = CC_all(stage_starts(s):stage_boundary(s));
    fprintf('  Stage%d %-5s [%-32s] | %10.2f -> %10.2f | Improvement %.2f%%\n', ...
        s, stage_names{s}, stage_labels{s}, seg(1), seg(end), ...
        100*(seg(1)-seg(end))/max(seg(1),1));
end

TED = bestFit5 + alldis_fixed;
total_pts = size(data.start,1);
ATD = TED / total_pts;

all_dists = zeros(total_pts,1);
for i=1:dim
    idx = max(Lb(i), min(X_final(i), Ub(i)));
    all_dists(i) = data.dis(DFenPei{i}(1), DFenPei{i}(idx+1));
end
for i=1:size(FID,1)
    all_dists(dim+i) = data.dis(FID(i,1), FID(i,2));
end
MID = max(all_dists);

dyn_bins = arrayfun(@(i) DFenPei{i}(min(X_final(i)+1, length(DFenPei{i}))), 1:dim);
SUR = length(unique([dyn_bins, FID(:,2)'])) / size(data.binan,1) * 100;
MET = toc;

fprintf('\n  [8 Evaluation KPIs]\n');
fprintf('  TED  (Total Evacuation Distance) : %.2f m\n',  TED);
fprintf('  ATD  (Average Trip Distance)     : %.2f m\n',  ATD);
fprintf('  MID  (Max Individual Distance)   : %.2f m\n',  MID);
fprintf('  SUR  (Shelter Utilization Rate)  : %.2f %%\n', SUR);
fprintf('  BTV  (Final Best Fitness)        : %.4f\n',    bestFit5);
fprintf('  MET  (Total Runtime)             : %.2f s\n',  MET);
fprintf('  Total Iterations                 : %d\n',      length(CC_all));
fprintf('========================================================\n');

%% ======================== Visualization ========================

C_W    = [252, 251, 248] / 255;   % off-white background
C_GRID = [0.88 0.88 0.85];        % subtle grid color

% ── shared gradient colormap (green → gold, 256 steps) ──────
n_c = 256; h2 = n_c/2;
cmap_gg = [ ...
    linspace(C_GREEN(1),0.97,h2)', linspace(C_GREEN(2),0.97,h2)', linspace(C_GREEN(3),0.97,h2)'; ...
    linspace(0.97,C_GOLD(1),h2)',  linspace(0.97,C_GOLD(2),h2)',  linspace(0.97,C_GOLD(3),h2)'];

% ── per-stage cell arrays for convenience ───────────────────
stage_segs  = cell(1,5);
stage_imprs = zeros(1,5);
for s=1:5
    stage_segs{s} = CC_all(stage_starts(s):stage_boundary(s));
    stage_imprs(s)= max(stage_segs{s}(1)-stage_segs{s}(end), 0);
end

%% ── Figure 1: Full Convergence Curve (5-stage coloured) ────
fig1 = figure('Name','Convergence Curve','Color',C_W,'Position',[50 50 1050 500]);
ax1  = axes('Color',C_W); hold on; box on; grid on;
ax1.GridColor = C_GRID; ax1.GridAlpha = 0.5;
ax1.XColor = [0.35 0.35 0.35]; ax1.YColor = [0.35 0.35 0.35];

for s=1:5
    sx = (stage_starts(s):stage_boundary(s))';
    sy = stage_segs{s};
    plot(sx, sy, 'Color', stage_colors{s}, 'LineWidth', 2.8, ...
         'DisplayName', sprintf('Stage %d: %s  [%s]', s, stage_names{s}, stage_labels{s}));
end
% Stage boundary markers
for s=1:4
    xb = stage_boundary(s);
    xline(xb,'--','Color',[0.65 0.65 0.60],'LineWidth',1.0,'HandleVisibility','off');
    yb = CC_all(xb);
    text(xb+length(CC_all)*0.005, yb, ...
         sprintf('\\rightarrow S%d',s+1), ...
         'FontSize',8,'Color',[0.40 0.40 0.40],'HandleVisibility','off');
end
% Annotate final value
yf = CC_all(end);
plot(length(CC_all), yf, 'p', 'MarkerSize',13, ...
     'MarkerFaceColor',C_GOLD,'MarkerEdgeColor','w', ...
     'LineWidth',0.8,'DisplayName',sprintf('Final: %.0f m',yf));

xlabel('Cumulative Iteration','FontWeight','bold','FontSize',11);
ylabel('Best Fitness — Total Distance (m)','FontWeight','bold','FontSize',11);
title('Pipeline Integrative Hybrid  (ABC \rightarrow NSGA\times4)', ...
      'FontSize',13,'FontWeight','bold','Color',[0.15 0.15 0.15]);
lgd = legend('Location','northeast','Box','off','FontSize',9);
set(ax1,'TickDir','out','FontSize',9);
exportgraphics(fig1,'pip_fig1_convergence.png','Resolution',200);
fprintf('Saved: pip_fig1_convergence.png\n');

%% ── Figure 2: Per-Stage Contribution Bar ───────────────────
fig2 = figure('Name','Stage Contribution','Color',C_W,'Position',[200 200 780 460]);
ax2  = axes('Color',C_W); hold on; box on; grid on;
ax2.GridColor = C_GRID; ax2.GridAlpha = 0.45;
ax2.XColor = [0.35 0.35 0.35]; ax2.YColor = [0.35 0.35 0.35];

bh2 = bar(stage_imprs,'FaceColor','flat','EdgeColor','none','BarWidth',0.62);
for s=1:5, bh2.CData(s,:) = stage_colors{s}; end

% Value labels
mx_i = max(stage_imprs);
for s=1:5
    text(s, stage_imprs(s)+mx_i*0.018, sprintf('%.0f m', stage_imprs(s)), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold', ...
        'Color',[0.2 0.2 0.2]);
end

% Mutation rate annotation
mut_rates = [NaN, 0.08, 0.15, 0.04, 0.02];
pop_sizes  = [FN1, popSz2, popSz3, popSz4, popSz5];
for s=1:5
    yl = -mx_i*0.07;
    if s==1
        lbl = sprintf('Pop=%d',pop_sizes(s));
    else
        lbl = sprintf('Pop=%d\nmut=%.2f',pop_sizes(s),mut_rates(s));
    end
    text(s, yl, lbl,'HorizontalAlignment','center','FontSize',8, ...
        'Color',stage_colors{s},'FontWeight','bold');
end

xlim([0.4 5.6]);
xticklabels({'S1: ABC','S2: NSGA','S3: NSGA','S4: NSGA','S5: NSGA'});
set(ax2,'FontSize',10,'TickDir','out');
xlabel('Algorithm Stage','FontWeight','bold','FontSize',11);
ylabel('Fitness Improvement (m)','FontWeight','bold','FontSize',11);
title('Per-Stage Contribution — Pipeline (ABC \rightarrow NSGA\times4)', ...
      'FontSize',12,'FontWeight','bold','Color',[0.15 0.15 0.15]);
exportgraphics(fig2,'pip_fig2_contribution.png','Resolution',200);
fprintf('Saved: pip_fig2_contribution.png\n');

%% ── Figure 3: Stage Summary Heatmap ────────────────────────
%  Shows start value, end value, improvement% and mutation rate
%  in a colour-coded 4×5 table.
fig3 = figure('Name','Stage Summary Heatmap','Color',C_W,'Position',[250 150 900 380]);
ax3  = axes('Position',[0.10 0.18 0.78 0.65],'Color',C_W);

row_labels = {'Start (m)','End (m)','Improv %','Mut Rate / Pop'};
nRow = 4; nCol = 5;

hmap3 = zeros(nRow, nCol);
for s=1:5
    seg = stage_segs{s};
    hmap3(1,s) = seg(1);
    hmap3(2,s) = seg(end);
    hmap3(3,s) = 100*(seg(1)-seg(end))/max(seg(1),1);
    if s==1
        hmap3(4,s) = FN1;          % show pop size for ABC
    else
        hmap3(4,s) = mut_rates(s);
    end
end

% Normalise each row 0-1 for colour mapping
hmap3_norm = zeros(nRow,nCol);
for r=1:nRow
    rmin=min(hmap3(r,:)); rmax=max(hmap3(r,:));
    if rmax-rmin<1e-10, hmap3_norm(r,:)=0.5;
    else, hmap3_norm(r,:)=(hmap3(r,:)-rmin)/(rmax-rmin); end
end
% Rows 1-2: lower=better → invert
hmap3_norm(1,:) = 1 - hmap3_norm(1,:);
hmap3_norm(2,:) = 1 - hmap3_norm(2,:);

imagesc(hmap3_norm);
colormap(ax3, cmap_gg);
try, clim([0 1]); catch, caxis([0 1]); end

% Text overlay
fmt = {'%.0f','%.0f','%.1f%%','%.3f'};
for r=1:nRow
    for c=1:5
        if r==4 && c==1
            lbl = sprintf('Pop=%d', int32(hmap3(4,1)));
        elseif r==4
            lbl = sprintf('%.3f', hmap3(r,c));
        else
            lbl = sprintf(fmt{r}, hmap3(r,c));
        end
        text(c,r,lbl,'HorizontalAlignment','center','VerticalAlignment','middle', ...
            'FontSize',9,'FontWeight','bold','Color',[0.12 0.12 0.12]);
    end
end

xticks(1:5);
xticklabels(cellfun(@(n,l) sprintf('S%d\n%s\n(%s)', ...
    find(strcmp(stage_names,n),1), n, l(1:min(12,end))), ...
    stage_names, stage_labels,'UniformOutput',false));
yticks(1:nRow); yticklabels(row_labels);
set(ax3,'TickDir','out','FontSize',9,'Box','off');

cb3 = colorbar('eastoutside','Position',[0.91 0.18 0.022 0.65]);
cb3.Label.String = 'Relative quality  (green=better, gold=worse)';
cb3.FontSize = 8; cb3.Ticks=[0 1];
cb3.TickLabels={'Better','Worse'};

title('Stage-by-Stage Summary Heatmap  (ABC \rightarrow NSGA\times4)', ...
    'FontSize',12,'FontWeight','bold','Color',[0.15 0.15 0.15]);
exportgraphics(fig3,'pip_fig3_summary_heatmap.png','Resolution',200);
fprintf('Saved: pip_fig3_summary_heatmap.png\n');

%% ── Figure 4: Stacked Area — Cumulative Improvement ────────
fig4 = figure('Name','Cumulative Improvement','Color',C_W,'Position',[300 100 980 430]);
ax4  = axes('Color',C_W); hold on; box on; grid on;
ax4.GridColor = C_GRID; ax4.GridAlpha = 0.45;
ax4.XColor = [0.35 0.35 0.35]; ax4.YColor = [0.35 0.35 0.35];

% Relative improvement from initial value
init_val = CC_all(1);
rel_impr  = max(init_val - CC_all, 0);   % monotone improvement

% Shade each stage
for s=1:5
    sx  = stage_starts(s):stage_boundary(s);
    sy1 = rel_impr(sx);
    fill([sx, fliplr(sx)],[sy1, zeros(1,length(sx))], ...
        stage_colors{s},'FaceAlpha',0.28,'EdgeColor','none');
    plot(sx, sy1,'Color',stage_colors{s},'LineWidth',2.2, ...
         'DisplayName',sprintf('S%d %s (%s)',s,stage_names{s},stage_labels{s}));
end

xlabel('Cumulative Iteration','FontWeight','bold','FontSize',11);
ylabel('Cumulative Improvement vs. Initial (m)','FontWeight','bold','FontSize',11);
title('Cumulative Improvement Breakdown by Stage','FontSize',12,'FontWeight','bold','Color',[0.15 0.15 0.15]);
legend('Location','northwest','Box','off','FontSize',9);
set(ax4,'TickDir','out','FontSize',9);
exportgraphics(fig4,'pip_fig4_cumulative_improvement.png','Resolution',200);
fprintf('Saved: pip_fig4_cumulative_improvement.png\n');

%% ── Figure 5: KPI Radar ─────────────────────────────────────
fig5 = figure('Name','KPI Radar','Color',C_W,'Position',[350 120 740 660]);
axes('Color',C_W); hold on;

kpi_names  = {'TED','ATD','MID','SUR','BTV Norm','Runtime'};
nKpi = length(kpi_names);

% Normalise KPIs to [0,1] for radar (higher=better after inversion)
kpi_raw   = [TED, ATD, MID, SUR, bestFit5, MET];
% For TED/ATD/MID/BTV/Runtime: lower=better → invert after normalisation
% For SUR: higher=better
kpi_norm  = zeros(1,nKpi);
kpi_ref   = [TED*1.2, ATD*1.2, MID*1.2, 100, bestFit5*1.3, MET*1.2];
kpi_base  = [0, 0, 0, 0, 0, 0];
for k=1:nKpi
    kpi_norm(k) = (kpi_raw(k)-kpi_base(k)) / max(kpi_ref(k)-kpi_base(k),1e-10);
end
% Invert lower-is-better metrics
invert_mask = [1,1,1,0,1,1];
kpi_norm(logical(invert_mask)) = 1 - kpi_norm(logical(invert_mask));
kpi_norm = max(0, min(1, kpi_norm));

theta = linspace(0, 2*pi, nKpi+1); theta(end)=[];

% Grid rings
for rv=[0.25,0.5,0.75,1.0]
    th_=linspace(0,2*pi,300);
    plot(cos(th_)*rv, sin(th_)*rv,'-','Color',C_GRID,'LineWidth',0.7);
    text(0, rv+0.04, sprintf('%.0f%%',rv*100),'FontSize',7, ...
        'HorizontalAlignment','center','Color',[0.6 0.6 0.6]);
end
for k=1:nKpi
    [xe,ye]=pol2cart(theta(k),1.0);
    plot([0 xe],[0 ye],'--','Color',C_GRID,'LineWidth',0.8);
    [xl_,yl_]=pol2cart(theta(k),1.28);
    text(xl_,yl_,kpi_names{k},'HorizontalAlignment','center', ...
        'FontSize',10,'FontWeight','bold','Color',[0.25 0.25 0.25]);
end

valsc=[kpi_norm,kpi_norm(1)]; thetac=[theta,theta(1)];
[xp,yp]=pol2cart(thetac,valsc);
fill(xp,yp,C_GREEN,'FaceAlpha',0.18,'EdgeColor','none');
plot(xp,yp,'-o','Color',C_GREEN,'LineWidth',2.5,'MarkerSize',8, ...
    'MarkerFaceColor',C_GREEN,'MarkerEdgeColor','w');

% Raw value annotations
for k=1:nKpi
    [xl_,yl_]=pol2cart(theta(k), kpi_norm(k)+0.09);
    if k==4
        lbl=sprintf('%.1f%%',SUR);
    elseif k==5
        lbl=sprintf('%.0f',bestFit5);
    elseif k==6
        lbl=sprintf('%.1fs',MET);
    elseif k==1
        lbl=sprintf('%.0fm',TED);
    elseif k==2
        lbl=sprintf('%.1fm',ATD);
    else
        lbl=sprintf('%.0fm',MID);
    end
    text(xl_,yl_,lbl,'HorizontalAlignment','center','FontSize',9, ...
        'Color',C_GOLD,'FontWeight','bold');
end
axis equal off;
title('Pipeline KPI Radar  (ABC \rightarrow NSGA\times4)', ...
    'FontSize',12,'FontWeight','bold','Color',[0.15 0.15 0.15]);
exportgraphics(fig5,'pip_fig5_kpi_radar.png','Resolution',200);
fprintf('Saved: pip_fig5_kpi_radar.png\n');

%% ── Figure 6: 2D Evacuation Allocation Map ──────────────────
fig6 = figure('Color',C_W,'Name','Allocation Map','Position',[100 100 900 820]);
ax6  = axes('Color',[0.96 0.97 0.96]);
hold on; box on;

% Road network (if available)
if isfield(data,'road')
    for i=1:length(data.road)
        plot(data.road{i}(:,1)-offset_x, data.road{i}(:,2)-offset_y, ...
             'Color',[0.88 0.88 0.88],'LineWidth',0.5);
    end
end

% Evacuation routes
for i=1:dim
    oi = max(1, min(X_final(i), length(DFenPei{i})-1));
    line([house_x(DFenPei{i}(1)), binan_x(DFenPei{i}(oi+1))], ...
         [house_y(DFenPei{i}(1)), binan_y(DFenPei{i}(oi+1))], ...
         'Color',[C_GREEN, 0.13],'LineWidth',0.4);
end
for i=1:size(FID,1)
    line([house_x(FID(i,1)), binan_x(FID(i,2))], ...
         [house_y(FID(i,1)), binan_y(FID(i,2))], ...
         'Color',[C_GREEN, 0.13],'LineWidth',0.4);
end

% Residential points
h_res = scatter(house_x, house_y, 12, ...
    'MarkerFaceColor',[0.20 0.40 0.75], ...
    'MarkerEdgeColor','none','MarkerFaceAlpha',0.7);

% Shelter points (sized by load)
shelter_load = zeros(size(data.binan,1),1);
for i=1:dim
    oi = max(1,min(X_final(i),length(DFenPei{i})-1));
    sid = DFenPei{i}(oi+1);
    shelter_load(sid) = shelter_load(sid) + 12;
end
for i=1:size(FID,1)
    shelter_load(FID(i,2)) = shelter_load(FID(i,2)) + 12;
end

% Colour shelters by load (green→gold gradient)
max_load = max(shelter_load,1);
for s_=1:size(data.binan,1)
    t_ = shelter_load(s_) / max_load;
    sc_ = (1-t_)*C_GREEN + t_*C_GOLD;
    scatter(binan_x(s_), binan_y(s_), ...
        max(50, shelter_load(s_)/max_load*160+30), ...
        sc_,'^','filled','MarkerEdgeColor','k','LineWidth',0.5);
end
h_shl = scatter(NaN,NaN,80,'g','^','filled','MarkerEdgeColor','k');

axis equal; axis tight;
ax6.XAxis.Exponent=0; ax6.YAxis.Exponent=0;
grid on; ax6.GridColor=C_GRID; ax6.GridAlpha=0.4;

title('Pipeline Optimized Evacuation Allocation Map  (ABC \rightarrow NSGA\times4)', ...
    'FontSize',12,'FontWeight','bold','Color',[0.15 0.15 0.15]);
legend([h_shl, h_res],{'Shelter (size ∝ load)','Residential'}, ...
    'Location','northeast','Box','off','FontSize',10);

% KPI text box
ann_str = sprintf(' TED=%.0fm  |  ATD=%.1fm  |  SUR=%.1f%%  |  MID=%.0fm', ...
    TED, ATD, SUR, MID);
annotation('textbox',[0.10 0.01 0.82 0.04],'String',ann_str, ...
    'FitBoxToText','off','EdgeColor','none','BackgroundColor',[0.95 0.95 0.92], ...
    'HorizontalAlignment','center','FontSize',9,'Color',[0.25 0.25 0.25]);

exportgraphics(fig6,'pip_fig6_allocation_map.png','Resolution',200);
fprintf('Saved: pip_fig6_allocation_map.png\n');

fprintf('\n[Done] Pipeline Integrative (ABC->NSGA×4) complete.\n');
fprintf('6 figures saved to current directory.\n');

%% ============================================================
%  Helper Functions
%% ============================================================

function fitness = unified_fobj(x, DFenPei, dis_mat, Lb, Ub)
    fitness=0;
    for i=1:length(DFenPei)
        idx=max(Lb(i), min(round(x(i)), Ub(i)));
        fitness=fitness+dis_mat(DFenPei{i}(1), DFenPei{i}(idx+1));
    end
end

function f2 = nsga_dual_obj(x, DFenPei, dis_mat, P_)
    X  = max(1, min(round(x), P_));
    td = 0;
    Y  = zeros(1, size(dis_mat,2));
    for i=1:length(X)
        hid=DFenPei{i}(1); eid=DFenPei{i}(X(i)+1);
        td=td+dis_mat(hid,eid);
        Y(eid)=Y(eid)+12;
    end
    f2=[td, var(Y)];
end

function pop = elite_inject(popSize, dim, Lb, Ub, elite_pop, elite_fit, K_elite)
    pop = zeros(popSize, dim);
    ne  = min(size(elite_pop,1), K_elite);
    for i=1:popSize
        if i<=ne && ne>0
            pop(i,:) = elite_pop(i,:);
        elseif ne>0
            base  = elite_pop(randi(ne),:);
            noise = round(randn(1,dim) .* max(1,(Ub-Lb)*0.03));
            pop(i,:) = max(Lb, min(Ub, base+noise));
        else
            pop(i,:) = Lb + round(rand(1,dim).*(Ub-Lb));
        end
    end
end