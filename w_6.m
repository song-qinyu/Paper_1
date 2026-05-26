%% ============================================================
%  Pipeline Hybrid — Stage Competition Framework v3
%  (All 15 algorithms + SHAP-style 7 figures, fully in English)
%
%  Changes vs original:
%    1. stage_labels -> English names matching the 5-stage image
%    2. Fig6: shows ALL 15 algorithms (3 rows x 5 cols), was 6
%    3. All Chinese text -> English (titles, labels, fprintf, colorbar)
%    4. Bug fixes retained: FontStyle removed, caxis/clim dual-write,
%       smooth fallback, rng_f/rng6 zero-guard, safe_set_ax as plain fn
%% ============================================================
clc; clear; close all; tic;

fprintf('========================================================\n');
fprintf('  Pipeline Stage Competition Framework v3\n');
fprintf('  Full Benchmark: 15 Algorithms\n');
fprintf('========================================================\n\n');

%% ==================== 0. Data Loading ====================
if exist('sj5.mat','file'), load('sj5.mat');
else, error('sj5.mat not found'); end

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

%% ==================== 1. Budget ====================
NFE_per_algo = 15000;
K_elite = 20;

%% ==================== 2. Five-Stage Competition ====================
elite_pop = []; elite_fit = [];
all_results = cell(5,1);

% ---- Stage labels: match the image exactly ----
stage_labels = {'Global Exploration', ...
                'Hierarchical Convergence', ...
                'Stagnation Breakout', ...
                'Neighborhood Intensification', ...
                'Deep Convergence'};

for stage = 1:5
    fprintf('--- Stage %d: %s ---\n', stage, stage_labels{stage});
    if stage==1
        fprintf('  (No elite input, fully random initialization)\n');
    else
        fprintf('  (Receiving elite pool of %d solutions from previous stage)\n', size(elite_pop,1));
    end
    results = run_stage_competition(fobj, dim, Lb, Ub, ...
        NFE_per_algo, elite_pop, elite_fit, K_elite);
    all_results{stage} = results;
    fprintf('\n[Stage %d Ranking]\n', stage);
    print_ranking(results);
    [elite_pop, elite_fit] = get_elite_pool(results, K_elite, dim);
    fprintf('\n');
end

%% ==================== 3. Summary Report ====================
fprintf('========================================================\n');
fprintf('  Per-Stage Best Algorithm Summary\n');
fprintf('========================================================\n');
for s=1:5
    r=all_results{s}; fits=[r.best_fit]; [~,wi]=min(fits);
    fprintf('  Stage %d (%s): Best=%-6s | Value=%.2f m\n',...
        s, stage_labels{s}, r(wi).name, r(wi).best_fit);
end
fprintf('\nSuggested Pipeline Combination:\n');
for s=1:5
    r=all_results{s}; [~,wi]=min([r.best_fit]);
    fprintf('  Stage %d -> %s\n', s, r(wi).name);
end
fprintf('\nTotal elapsed time: %.2f s\n', toc);
fprintf('========================================================\n');

%% ==================== 4. Build Data Matrix ====================
algos_all = {'GA','DE','PSO','SSA','GWO','FA','ABC','TOW','CA','PO','CS','HLO','SA','HS','NSGA'};
nAlgo  = length(algos_all);
nStage = 5;
all_fit = nan(nAlgo, nStage);

for s = 1:nStage
    r = all_results{s};
    for a = 1:length(r)
        idx = find(strcmp(algos_all, r(a).name));
        if ~isempty(idx)
            all_fit(idx, s) = r(a).best_fit;
        end
    end
end
for s=1:nStage
    col=all_fit(:,s); m=nanmean(col);
    col(isnan(col))=m; all_fit(:,s)=col;
end

%% ==================== 5. Visualization (7 figures) ====================
pipeline_viz(all_fit, stage_labels, algos_all);


%% ============================================================
%%  Visualization Main Function
%% ============================================================
function pipeline_viz(all_fit, stage_labels, algos)

nAlgo  = size(all_fit,1);
nStage = size(all_fit,2);

C_GREEN = [88,  140, 90]  /255;
C_GOLD  = [214, 164, 59]  /255;
C_LG    = [160, 200, 130] /255;
C_LY    = [240, 210, 140] /255;
C_W     = [252, 251, 248] /255;

n_c=256; h2=n_c/2;
cmap_gg = [linspace(C_GREEN(1),0.97,h2)', linspace(C_GREEN(2),0.97,h2)', linspace(C_GREEN(3),0.97,h2)';
           linspace(0.97,C_GOLD(1),h2)',  linspace(0.97,C_GOLD(2),h2)',  linspace(0.97,C_GOLD(3),h2)'];

%% ===== Fig 1: Per-Stage Ranking Bar Charts =====
fig1 = figure('Color',C_W,'Name','fig1','Position',[30 30 1600 900]);
for s = 1:nStage
    ax = subplot(2,3,s);
    fits = all_fit(:,s);
    [fits_s, si] = sort(fits,'ascend');
    names_s = algos(si);
    n = length(fits_s);
    rng_f = max(fits_s)-min(fits_s);
    if rng_f < 1, rng_f=1; end
    f_norm = (fits_s-min(fits_s))/rng_f;
    colors = zeros(n,3);
    for i=1:n
        colors(i,:) = (1-f_norm(i))*C_GREEN + f_norm(i)*C_GOLD;
    end
    bh = barh(fits_s,'FaceColor','flat','EdgeColor','none','BarWidth',0.68);
    for i=1:n, bh.CData(i,:)=colors(i,:); end
    hold on;
    plot(fits_s(1), 1,'p','MarkerFaceColor',C_GREEN,...
        'MarkerEdgeColor','w','MarkerSize',11,'LineWidth',0.5);
    for i=1:n
        text(fits_s(i)+rng_f*0.005, i, sprintf('%.0f',fits_s(i)),...
            'FontSize',7.5,'Color',[0.3 0.3 0.3],'VerticalAlignment','middle');
    end
    yticks(1:n); yticklabels(names_s);
    xlabel('Total Distance (m)','FontSize',9);
    title(sprintf('Stage %d  .  %s', s, stage_labels{s}),...
        'FontSize',10,'FontWeight','bold','Color',[0.2 0.2 0.2]);
    set(ax,'Color',C_W,'Box','off','TickDir','out','FontSize',9,...
        'GridAlpha',0.12,'XColor',[0.35 0.35 0.35],'YColor',[0.35 0.35 0.35]);
    grid on;
end

ax6 = subplot(2,3,6);
stage_bests   = min(all_fit,[],1);
stage_winners = cell(1,nStage);
for s=1:nStage
    [~,wi]=min(all_fit(:,s)); stage_winners{s}=algos{wi};
end
rng6 = max(stage_bests)-min(stage_bests);
if rng6<1, rng6=1; end
fill([1:nStage,fliplr(1:nStage)],...
    [stage_bests+rng6*0.05, fliplr(stage_bests-rng6*0.05)],...
    C_LG,'EdgeColor','none','FaceAlpha',0.35); hold on;
plot(1:nStage, stage_bests,'-o','LineWidth',2.5,'Color',C_GREEN,...
    'MarkerFaceColor',C_GREEN,'MarkerSize',8,'MarkerEdgeColor','w');
for s=1:nStage
    text(s, stage_bests(s)+rng6*0.07, stage_winners{s},...
        'HorizontalAlignment','center','FontSize',9,...
        'Color',C_GREEN,'FontWeight','bold');
end
xlim([0.6 nStage+0.4]); xticks(1:nStage);
xticklabels(cellfun(@(l,k) sprintf('S%d\n%s',k,l),...
    stage_labels,num2cell(1:nStage),'UniformOutput',false));
ylabel('Best Fitness (m)','FontSize',9);
title('Pipeline Convergence Trend','FontSize',11,'FontWeight','bold','Color',[0.2 0.2 0.2]);
set(ax6,'Color',C_W,'Box','off','TickDir','out','FontSize',9,...
    'GridAlpha',0.12,'XColor',[0.35 0.35 0.35],'YColor',[0.35 0.35 0.35]);
grid on;
sgtitle('Per-Stage Algorithm Competition Ranking  (Lower Distance = Better)',...
    'FontSize',12,'FontWeight','bold','Color',[0.15 0.15 0.15]);
exportgraphics(fig1,'fig1_stage_ranking.png','Resolution',200);
fprintf('Saved: fig1_stage_ranking.png\n');

%% ===== SHAP Matrix Calculation =====
shap_mat = zeros(nAlgo,nStage);
for s=1:nStage
    mu=mean(all_fit(:,s));
    shap_mat(:,s)=(mu-all_fit(:,s))/mu;
end
mean_shap = mean(abs(shap_mat),2);

%% ===== Fig 2: SHAP Beeswarm =====
[mean_shap_s, si_shap] = sort(mean_shap,'ascend');
algos_s = algos(si_shap);
shap_s  = shap_mat(si_shap,:);

fig2 = figure('Color',C_W,'Name','fig2','Position',[60 60 920 640]);
ax2 = axes('Color',C_W); hold on;
for i=1:nAlgo
    y_vals=shap_s(i,:);
    barh(i, mean_shap_s(i), 0.4,...
        'FaceColor',C_LG.*0.7+C_W*0.3,'EdgeColor','none','FaceAlpha',0.5);
    for j=1:nStage
        yj=i+(rand-0.5)*0.35;
        v=max(-0.12,min(0.12,y_vals(j)));
        t=(v+0.12)/0.24;
        c=(1-t)*C_GREEN+t*C_GOLD;
        scatter(abs(y_vals(j)),yj,45,c,'filled',...
            'MarkerFaceAlpha',0.82,'MarkerEdgeColor','none');
    end
    pct=mean_shap_s(i)/sum(mean_shap_s)*100;
    text(mean_shap_s(i)+max(mean_shap_s)*0.01, i,...
        sprintf(' %.1f%%',pct),'FontSize',8,...
        'Color',[0.3 0.3 0.3],'VerticalAlignment','middle');
end
yticks(1:nAlgo); yticklabels(algos_s);
xlabel('Mean |SHAP| (Relative Contribution)','FontSize',10);
title('SHAP Global Algorithm Contribution (Beeswarm)','FontSize',12,'FontWeight','bold','Color',[0.2 0.2 0.2]);
colormap(ax2,[linspace(C_GREEN(1),C_GOLD(1),64)',...
              linspace(C_GREEN(2),C_GOLD(2),64)',...
              linspace(C_GREEN(3),C_GOLD(3),64)']);
try, clim([0 1]); catch, caxis([0 1]); end
cb2=colorbar('eastoutside');
cb2.Label.String='Stage Fitness  Low -> High';
cb2.Ticks=[0,1]; cb2.TickLabels={'Poor (negative)','Good (positive)'}; cb2.FontSize=8;
set(ax2,'Color',C_W,'Box','off','TickDir','out','FontSize',9,'GridAlpha',0.12,...
    'XColor',[0.35 0.35 0.35],'YColor',[0.35 0.35 0.35]);
grid on; ax2.XGrid='on'; ax2.YGrid='off';
exportgraphics(fig2,'fig2_shap_beeswarm.png','Resolution',200);
fprintf('Saved: fig2_shap_beeswarm.png\n');

%% ===== Fig 3: Main Effect vs Interaction Effect =====
main_eff  = mean(abs(shap_mat),2);
inter_eff = std(shap_mat,0,2)*0.5;
[~,si3]   = sort(main_eff,'descend');

fig3 = figure('Color',C_W,'Name','fig3','Position',[90 50 1150 430]);
ax3 = axes('Color',C_W); hold on;
bw3=0.36;
bm  = bar((1:nAlgo)-bw3/2, main_eff(si3),  bw3,...
    'FaceColor',C_GREEN,'EdgeColor','none','FaceAlpha',0.88);
bi3 = bar((1:nAlgo)+bw3/2, inter_eff(si3), bw3,...
    'FaceColor',C_GOLD,'EdgeColor','none','FaceAlpha',0.88);
mx_m=max(main_eff); mx_i=max(inter_eff);
for k=1:nAlgo
    text(k-bw3/2, main_eff(si3(k))+mx_m*0.015,...
        sprintf('%.3f',main_eff(si3(k))),'FontSize',7,...
        'HorizontalAlignment','center','Color',[0.2 0.2 0.2]);
    text(k+bw3/2, inter_eff(si3(k))+mx_i*0.015,...
        sprintf('%.3f',inter_eff(si3(k))),'FontSize',7,...
        'HorizontalAlignment','center','Color',[0.35 0.25 0.05]);
end
xticks(1:nAlgo); xticklabels(algos(si3)); xtickangle(35);
ylabel('Magnitude (Mean |SHAP|)','FontSize',10);
title('Main Effect vs. Interaction Effect','FontSize',12,'FontWeight','bold','Color',[0.2 0.2 0.2]);
legend([bm,bi3],{'Main Effect (Mean |SHAP|)','Interaction (Std over Stages)'},...
    'Location','northeast','FontSize',9,'Box','off');
set(ax3,'Color',C_W,'Box','off','TickDir','out','FontSize',9,'GridAlpha',0.12,...
    'XColor',[0.35 0.35 0.35],'YColor',[0.35 0.35 0.35]);
grid on; ax3.XGrid='off';
exportgraphics(fig3,'fig3_main_vs_interaction.png','Resolution',200);
fprintf('Saved: fig3_main_vs_interaction.png\n');

%% ===== Fig 4: SHAP Heatmap =====
[~,row_order] = sort(mean(abs(shap_mat),2),'descend');
hmap = shap_mat(row_order,:);

fig4 = figure('Color',C_W,'Name','fig4','Position',[120 40 960 560]);
ax4 = axes('Position',[0.12 0.11 0.72 0.72],'Color',C_W);
imagesc(hmap);
colormap(ax4, cmap_gg);
clim_v=max(abs(hmap(:)));
if clim_v<1e-10, clim_v=1; end
try, clim([-clim_v clim_v]); catch, caxis([-clim_v clim_v]); end

cb4=colorbar('eastoutside','Position',[0.87 0.11 0.025 0.72]);
cb4.Label.String='SHAP Value'; cb4.FontSize=8;
cb4.Ticks=[-clim_v,0,clim_v];
cb4.TickLabels={sprintf('-%.3f',clim_v),'0',sprintf('+%.3f',clim_v)};

for i=1:nAlgo
    for j=1:nStage
        text(j,i,sprintf('%.3f',hmap(i,j)),...
            'HorizontalAlignment','center','VerticalAlignment','middle',...
            'FontSize',8.5,'Color',[0.15 0.15 0.15],'FontWeight','bold');
    end
end
xticks(1:nStage);
xticklabels(cellfun(@(l,k) sprintf('S%d %s',k,l),...
    stage_labels,num2cell(1:nStage),'UniformOutput',false));
yticks(1:nAlgo); yticklabels(algos(row_order));
xlabel('Pipeline Stage','FontSize',10);
title('SHAP Heatmap  (Algorithm x Stage Contribution Matrix)',...
    'FontSize',12,'FontWeight','bold','Color',[0.2 0.2 0.2]);
set(ax4,'TickDir','out','FontSize',9,'Box','off');

% Top f(x) trend line
ax4t = axes('Position',[0.12 0.845 0.72 0.10],'Color',C_W);
sb2  = min(all_fit,[],1);
rng4 = max(sb2)-min(sb2); if rng4<1, rng4=1; end
patch([1:nStage,fliplr(1:nStage)],...
    [sb2+rng4*0.08, fliplr(sb2-rng4*0.08)],...
    [0.75 0.75 0.72],'EdgeColor','none','FaceAlpha',0.4); hold on;
plot(1:nStage, sb2,'k-','LineWidth',1.5);
scatter(1:nStage, sb2, 30,'k','filled');
xlim([0.5 nStage+0.5]); axis off;
text(0.3, mean(sb2),'f(x)','FontSize',9,'Color',[0.3 0.3 0.3]);

exportgraphics(fig4,'fig4_shap_heatmap.png','Resolution',200);
fprintf('Saved: fig4_shap_heatmap.png\n');

%% ===== Fig 5: Algorithm Pairwise Interaction Matrix =====
cross_mat = zeros(nAlgo,nAlgo);
for i=1:nAlgo
    for j=1:nAlgo
        if i~=j
            cross_mat(i,j)=mean(all_fit(j,:)-all_fit(i,:));
        end
    end
end
fig5 = figure('Color',C_W,'Name','fig5','Position',[150 30 1050 900]);
ax5 = axes('Color',C_W);
imagesc(cross_mat);
colormap(ax5, cmap_gg);
mx5=max(abs(cross_mat(:)));
if mx5<1, mx5=1; end
try, clim([-mx5 mx5]); catch, caxis([-mx5 mx5]); end
cb5=colorbar;
cb5.Label.String='Performance Gap (m,  positive = row beats column)';
cb5.FontSize=8;
for i=1:nAlgo
    for j=1:nAlgo
        v=cross_mat(i,j);
        if i~=j && abs(v)>mx5*0.15
            text(j,i,sprintf('%.0f',v),...
                'HorizontalAlignment','center','VerticalAlignment','middle',...
                'FontSize',7,'Color',[0.15 0.15 0.15]);
        end
    end
end
xticks(1:nAlgo); xticklabels(algos); xtickangle(38);
yticks(1:nAlgo); yticklabels(algos);
xlabel('Reference Algorithm (Column)','FontSize',10);
ylabel('Target Algorithm (Row)','FontSize',10);
title('Algorithm Pairwise Interaction Matrix','FontSize',12,'FontWeight','bold','Color',[0.2 0.2 0.2]);
set(ax5,'TickDir','out','FontSize',9,'Box','off');
exportgraphics(fig5,'fig5_algo_interaction_matrix.png','Resolution',200);
fprintf('Saved: fig5_algo_interaction_matrix.png\n');

%% ===== Fig 6: SHAP Dependence — ALL 15 Algorithms (3 rows x 5 cols) =====
[~,top_all_idx] = sort(mean(abs(shap_mat),2),'descend');
top_all = top_all_idx(1:nAlgo);   % all 15 algorithms

nRows6 = 3; nCols6 = 5;           % 3 x 5 = 15 subplots
fig6 = figure('Color',C_W,'Name','fig6','Position',[40 40 1900 1100]);
for k=1:length(top_all)
    ax = subplot(nRows6, nCols6, k);
    a_idx=top_all(k);
    x_vals=all_fit(a_idx,:);
    y_vals=shap_mat(a_idx,:);
    yl_v=max(abs(y_vals))*1.35+0.005;
    rng_x=max(x_vals)-min(x_vals); if rng_x<1, rng_x=1; end
    xl_v=[min(x_vals)-rng_x*0.08, max(x_vals)+rng_x*0.08];

    patch([xl_v(1) xl_v(2) xl_v(2) xl_v(1)],[0 0 yl_v yl_v],...
        C_LY,'FaceAlpha',0.18,'EdgeColor','none'); hold on;
    patch([xl_v(1) xl_v(2) xl_v(2) xl_v(1)],[-yl_v -yl_v 0 0],...
        C_LG,'FaceAlpha',0.18,'EdgeColor','none');

    stage_cols=interp1([0,1],[C_GREEN;C_GOLD],linspace(0,1,nStage));
    for j=1:nStage
        xj=x_vals(j)+(rand-0.5)*rng_x*0.03;
        scatter(xj,y_vals(j),65,stage_cols(j,:),'filled',...
            'MarkerFaceAlpha',0.85,'MarkerEdgeColor','w','LineWidth',0.4);
    end

    [xs,si_]=sort(x_vals); ys_raw=y_vals(si_);
    fit_ok=false;
    if nStage>=4
        try
            ys_sm=smooth(xs(:),ys_raw(:),0.8,'lowess')';
            plot(xs,ys_sm,'-','LineWidth',2.2,'Color',C_GREEN);
            fit_ok=true;
        catch
        end
    end
    if ~fit_ok
        p=polyfit(x_vals,y_vals,1);
        xf=linspace(xl_v(1),xl_v(2),50);
        plot(xf,polyval(p,xf),'-','LineWidth',2.2,'Color',C_GREEN);
    end
    yline(0,'--','Color',[0.65 0.65 0.65],'LineWidth',0.9);
    xlim(xl_v); ylim([-yl_v yl_v]);
    xlabel('Fitness (m)','FontSize',8);
    ylabel('SHAP Value','FontSize',8);
    title(algos{a_idx},'FontSize',10,'FontWeight','bold','Color',[0.2 0.2 0.2]);
    set(ax,'Color',C_W,'Box','off','TickDir','out','FontSize',8,'GridAlpha',0.08,...
        'XColor',[0.35 0.35 0.35],'YColor',[0.35 0.35 0.35]);
    grid on;
end
sgtitle('SHAP Dependence Plots — All 15 Algorithms (Lowess Fit)',...
    'FontSize',13,'FontWeight','bold','Color',[0.15 0.15 0.15]);
exportgraphics(fig6,'fig6_shap_dependence.png','Resolution',200);
fprintf('Saved: fig6_shap_dependence.png\n');

%% ===== Fig 7: Top 5 Algorithms Radar Chart =====
score_mat=zeros(nAlgo,nStage);
for s=1:nStage
    [~,si_]=sort(all_fit(:,s),'ascend');
    rnk=zeros(nAlgo,1); rnk(si_)=(nAlgo:-1:1)';
    score_mat(:,s)=(rnk-1)/(nAlgo-1);
end
[~,top5]=sort(mean(score_mat,2),'descend');
top5=top5(1:min(5,nAlgo));

fig7=figure('Color',C_W,'Name','fig7','Position',[200 80 860 720]);
axes('Color',C_W); hold on;
theta=linspace(0,2*pi,nStage+1); theta(end)=[];
t5c={C_GREEN,C_GOLD,C_LG,[0.72 0.42 0.32],[0.38 0.55 0.75]};

for r_=[0.25,0.5,0.75,1.0]
    th_=linspace(0,2*pi,200);
    plot(cos(th_)*r_,sin(th_)*r_,'-','Color',[0.88 0.88 0.85],'LineWidth',0.6);
end
for s=1:nStage
    [xe,ye]=pol2cart(theta(s),1.0);
    plot([0 xe],[0 ye],'--','Color',[0.82 0.82 0.80],'LineWidth',0.7);
    [xl_,yl_]=pol2cart(theta(s),1.25);
    text(xl_,yl_,sprintf('S%d\n%s',s,stage_labels{s}),...
        'HorizontalAlignment','center','FontSize',9,...
        'Color',[0.28 0.28 0.28],'FontWeight','bold');
end
for k=1:length(top5)
    vals=score_mat(top5(k),:);
    valsc=[vals,vals(1)]; thetac=[theta,theta(1)];
    [xp,yp]=pol2cart(thetac,valsc);
    fill(xp,yp,t5c{k},'FaceAlpha',0.09,'EdgeColor','none');
    plot(xp,yp,'-o','Color',t5c{k},'LineWidth',2.2,'MarkerSize',6,...
        'MarkerFaceColor',t5c{k},'MarkerEdgeColor','w');
end
legend(algos(top5),'Location','southoutside','Orientation','horizontal',...
    'FontSize',9.5,'Box','off');
axis equal off;
title('Top 5 Algorithms — Stage Composite Score Radar',...
    'FontSize',12,'FontWeight','bold','Color',[0.2 0.2 0.2]);
exportgraphics(fig7,'fig7_radar.png','Resolution',200);
fprintf('Saved: fig7_radar.png\n');
fprintf('\nAll 7 figures saved to current directory.\n');

end  % pipeline_viz


%% ============================================================
%%  Core Scheduling Functions
%% ============================================================
function results=run_stage_competition(fobj,dim,Lb,Ub,NFE,elite_pop,elite_fit,K_elite)
    idx=0;
    results=struct('name',{},'best_fit',{},'best_pos',{},'pop',{},'fit',{});
    algo_list={
        'GA',   @run_GA;
        'DE',   @run_DE;
        'PSO',  @run_PSO;
        'SSA',  @run_SSA;
        'GWO',  @run_GWO;
        'FA',   @run_FA;
        'ABC',  @run_ABC;
        'TOW',  @run_TOW;
        'CA',   @run_CA;
        'PO',   @run_PO;
        'CS',   @run_CS;
        'HLO',  @run_HLO;
        'SA',   @run_SA;
        'HS',   @run_HS;
        'NSGA', @run_NSGA;
    };
    for a=1:size(algo_list,1)
        name=algo_list{a,1}; fn=algo_list{a,2};
        fprintf('  Evaluating %-4s ... ', name);
        try
            [bf,bp,pop_out,fit_out]=fn(fobj,dim,Lb,Ub,NFE,elite_pop,elite_fit,K_elite);
            idx=idx+1;
            results(idx).name=name; results(idx).best_fit=bf;
            results(idx).best_pos=bp; results(idx).pop=pop_out; results(idx).fit=fit_out;
            fprintf('Best: %.2f\n', bf);
        catch ME
            fprintf('[Skipped - Error] %s\n', ME.message);
        end
    end
end

function print_ranking(results)
    [fits_s,si]=sort([results.best_fit]);
    fprintf('  Rank | Algo  | Best Fitness\n');
    fprintf('  -----|-------|--------------------\n');
    for r=1:length(si)
        mark=''; if r==1, mark=' <- WINNER'; end
        fprintf('  %3d  | %-4s  | %.2f m%s\n',r,results(si(r)).name,fits_s(r),mark);
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
    pop=zeros(popSize,dim);
    ne=min(size(elite_pop,1),K_elite);
    for i=1:popSize
        if i<=ne&&ne>0
            pop(i,:)=elite_pop(i,:);
        elseif ne>0
            base=elite_pop(randi(ne),:);
            noise=round(randn(1,dim).*max(1,(Ub-Lb)*0.04));
            pop(i,:)=max(Lb,min(Ub,base+noise));
        else
            pop(i,:)=Lb+round(rand(1,dim).*(Ub-Lb));
        end
    end
end

%% ============================================================
%%  15 Algorithm Implementations
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

function [best_fit,best_pos,pop,fit]=run_DE(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    popSize=50; maxGen=floor(NFE/popSize); F=0.7; CR=0.8;
    pop=init_pop(popSize,dim,Lb,Ub,ep,K);
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

function [best_fit,best_pos,pop,fit]=run_PSO(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    popSize=50; maxIter=floor(NFE/popSize); w=0.8; c1=1.5; c2=1.5;
    pop=init_pop(popSize,dim,Lb,Ub,ep,K);
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
            if f<As, Ds=Bs;Dp=Bp;Bs=As;Bp=Ap;As=f;Ap=np;
            elseif f<Bs, Ds=Bs;Dp=Bp;Bs=f;Bp=np;
            elseif f<Ds, Ds=f;Dp=np; end
        end
        if As<best_fit, best_fit=As; best_pos=Ap; end
    end
end

function [best_fit,best_pos,pop,fit]=run_FA(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    n=40; maxIter=floor(NFE/n); alpha=0.5; betamin=0.2; gamma=1;
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
                    pop(i,:)=max(Lb,min(Ub,pop(i,:)));
                    fit(i)=fobj(pop(i,:));
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

function [best_fit,best_pos,pop,fit]=run_PO(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    parties=10; areas=10; popSize=parties*areas;
    maxIter=floor(NFE/popSize);
    pop=init_pop(popSize,dim,Lb,Ub,ep,K);
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
                    base=ep(randi(size(ep,1)),:);
                    np=max(Lb,min(Ub,base+randi([-10,10],1,dim)));
                else
                    np=Lb+round(rand(1,dim).*(Ub-Lb));
                end
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

function [best_fit,best_pos,pop,fit]=run_HLO(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    popSize=40; bpv=8; m=dim*bpv;
    maxIter=floor(NFE/popSize); p_r=0.1; p_i=0.5;
    bin_pop=zeros(popSize,m);
    for i=1:popSize
        if i<=min(size(ep,1),K)&&size(ep,1)>0
            for v=1:dim
                ratio=(double(ep(i,v))-Lb(v))/(max(Ub(v)-Lb(v),1));
                int_val=round(ratio*(2^bpv-1));
                bits=dec2bin(int_val,bpv)-'0';
                bin_pop(i,(v-1)*bpv+1:v*bpv)=bits;
            end
        else
            bin_pop(i,:)=randi([0,1],1,m);
        end
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
        DFenPei_=evalin('base','DFenPei');
        dis_mat_=evalin('base','data.dis');
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
    best_fit=fobj(x_int1); NFE_used=NFE_used+1; best_pos=x_int1;
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
                if f2<best_fit, best_fit=f2; best_pos=x_int2; end
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

function x_int=sa_to_fobj_x(sel_centers,DFenPei_,Lb,Ub)
    dim_=length(DFenPei_); x_int=zeros(1,dim_);
    for i=1:dim_
        cands=DFenPei_{i}(2:end);
        overlap=intersect(cands,sel_centers);
        if ~isempty(overlap)
            idx=find(DFenPei_{i}==overlap(1))-1;
        else
            idx=1;
        end
        x_int(i)=max(Lb(i),min(Ub(i),idx));
    end
end

function [best_fit,best_pos,pop,fit]=run_HS(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    try
        DFenPei_=evalin('base','DFenPei');
        data_=evalin('base','data');
        binan_xy=data_.binan; start_xy=data_.start;
    catch
        pop=init_pop(20,dim,Lb,Ub,ep,K);
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
        hm_fit(i)=hs_eval(HM(i,:),num_centers,house_x,house_y,fobj,DFenPei_,binan_xy,Lb,Ub);
        NFE_used=NFE_used+1;
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
    for i=2:pop_size
        pop(i,:)=max(Lb,min(Ub,best_pos+round(randn(1,dim))));
    end
    fit=arrayfun(@(i) fobj(pop(i,:)),1:pop_size)';
end

function fv=hs_eval(harm,nc,hx,hy,fobj,DFenPei_,binan_xy,Lb,Ub)
    x_int=hs_harm_to_int(harm,nc,DFenPei_,binan_xy,Lb,Ub); fv=fobj(x_int);
end

function x_int=hs_harm_to_int(harm,nc,DFenPei_,binan_xy,Lb,Ub)
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
        if ~isempty(overlap)
            idx=find(DFenPei_{i}==overlap(1))-1;
        else
            idx=1;
        end
        x_int(i)=max(Lb(i),min(Ub(i),idx));
    end
end

function [best_fit,best_pos,pop,fit]=run_NSGA(fobj,dim,Lb,Ub,NFE,ep,ef,K)
    try
        DFenPei_=evalin('base','DFenPei');
        data_=evalin('base','data'); dis_mat_=data_.dis;
    catch
        pop=init_pop(20,dim,Lb,Ub,ep,K);
        fit=arrayfun(@(i) fobj(pop(i,:)),1:20)';
        [best_fit,bi]=min(fit); best_pos=pop(bi,:); return;
    end
    popSize=min(100,floor(NFE/20)); maxGen=floor(NFE/popSize);
    P_=cellfun(@(x) length(x)-1,DFenPei_);
    nsga_obj=@(x) nsga_eval(x,DFenPei_,dis_mat_,P_);
    if ~isempty(ep)
        pop=init_pop(popSize,dim,Lb,Ub,ep,K);
    else
        pop=Lb+round(rand(popSize,dim).*(Ub-Lb));
    end
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
        combined=[pop;child];
        Fc=zeros(2*popSize,2);
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