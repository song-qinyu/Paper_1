%% ============================================================
%  Parallel Hybrid Algorithm — Concurrent Evolution Layer
%  Algorithms: NSGA · ABC · HLO · GWO · TOW
%
%  Architecture (as shown in diagram):
%    - 5 algorithms run CONCURRENTLY (parfor / for)
%    - A shared "Hub" broadcasts the global best solution
%      back to each algorithm at every sync checkpoint
%    - Each algo can ACCEPT the hub solution if it beats
%      its own current best  →  real-time elite exchange
%
%  Sync Strategy:
%    - NFE_total is divided into N_SYNC epochs
%    - After each epoch all 5 algos report their best
%    - Hub picks the global best → pushes back to all
%    - Each algo seeds its next epoch from the hub best
%
%  Output:
%    - hub_best_fit   : scalar, best fitness found
%    - hub_best_pos   : 1×dim, best solution vector
%    - sync_history   : N_SYNC × 5 matrix of per-algo bests
%    - conv_hybrid    : convergence curve (N_SYNC points)
%    - Saves: hybrid_fig1_convergence.png
%             hybrid_fig2_contribution.png
%             hybrid_fig3_sync_heatmap.png
%% ============================================================
clc; clear; close all; tic;

fprintf('============================================================\n');
fprintf('  Parallel Hybrid — Concurrent Evolution Layer\n');
fprintf('  NSGA | ABC | HLO | GWO | TOW\n');
fprintf('============================================================\n\n');

%% ==================== 0. Data Loading ====================
if exist('sj5.mat','file'), load('sj5.mat');
else, error('sj5.mat not found. Place it in the working directory.'); end

if exist('dis','var'), data.dis = dis; end

dim = length(DFenPei);
Lb  = ones(1, dim);
Ub  = arrayfun(@(i) length(DFenPei{i})-1, 1:dim);

fobj = @(x) unified_fobj(x, DFenPei, data.dis, Lb, Ub);

%% ==================== 1. Hybrid Settings ====================
NFE_TOTAL  = 75000;   % total function evaluations (same as benchmark)
N_SYNC     = 20;      % number of sync / exchange checkpoints
N_RUNS     = 5;       % independent hybrid runs
NFE_EPOCH  = floor(NFE_TOTAL / N_SYNC);   % NFE per algo per epoch

ALGO_NAMES = {'NSGA','ABC','HLO','GWO','TOW'};
N_ALGO     = length(ALGO_NAMES);

% Colour palette (matching your benchmark code)
C_GREEN = [88,  140,  90] /255;
C_GOLD  = [214, 164,  59] /255;
C_W     = [252, 251, 248] /255;
C_MID   = 0.5*C_GREEN + 0.5*C_GOLD;

% 5-step gold-green gradient: deep green → olive → warm mid → amber → deep gold
% Matches the green-gold colormap palette of Fig8 & Fig10
ALGO_COLORS = {
    [88,  140,  90] /255,    % NSGA  — deep green
    [138, 174,  90] /255,    % ABC   — olive green
    [178, 168,  80] /255,    % HLO   — olive gold (mid)
    [204, 158,  60] /255,    % GWO   — amber
    [214, 140,  45] /255     % TOW   — deep gold
};

n_c = 256; h2 = n_c/2;
cmap_gg = [ ...
    linspace(C_GREEN(1),0.97,h2)', linspace(C_GREEN(2),0.97,h2)', linspace(C_GREEN(3),0.97,h2)'; ...
    linspace(0.97,C_GOLD(1),h2)',  linspace(0.97,C_GOLD(2),h2)',  linspace(0.97,C_GOLD(3),h2)'];

fprintf('NFE total        : %d\n', NFE_TOTAL);
fprintf('Sync checkpoints : %d  (every %d NFE)\n', N_SYNC, NFE_EPOCH);
fprintf('Independent runs : %d\n', N_RUNS);
fprintf('Algorithms       : %s\n\n', strjoin(ALGO_NAMES,' | '));

%% ==================== 2. Parallel Pool ====================
use_parallel = ~isempty(ver('parallel'));
if use_parallel
    try
        pool = gcp('nocreate');
        if isempty(pool), parpool('local'); end
        fprintf('[INFO] Parallel pool active — using parfor for runs.\n\n');
    catch
        use_parallel = false;
        fprintf('[WARN] Could not start parallel pool — falling back to for-loop.\n\n');
    end
else
    fprintf('[INFO] Parallel Computing Toolbox not found — sequential mode.\n\n');
end

%% ==================== 3. Multi-Run Hybrid Evaluation ====================
% Storage across runs
all_hub_fit   = nan(1, N_RUNS);
all_hub_pos   = cell(1, N_RUNS);
all_sync_hist = nan(N_RUNS, N_SYNC, N_ALGO);   % run × sync × algo
all_conv      = nan(N_RUNS, N_SYNC);            % run × sync (hub best)
all_accept    = nan(N_RUNS, N_SYNC, N_ALGO);   % hub-injection accepted?

fprintf('%-6s | %-12s | %-12s | %-30s\n', 'Run','Hub_Best','Elapsed(s)','Winner-per-Sync (algo)');
fprintf('%s\n', repmat('-',1,70));

DFenPei_snap = DFenPei; data_snap = data;
Lb_snap = Lb; Ub_snap = Ub; dim_snap = dim;

for run_id = 1:N_RUNS
    t_run = tic;

    % ── Initialise each algorithm's state ──────────────────
    states = cell(1, N_ALGO);
    states{1} = nsga_init  (fobj, dim_snap, Lb_snap, Ub_snap, DFenPei_snap, data_snap);
    states{2} = abc_init   (fobj, dim_snap, Lb_snap, Ub_snap);
    states{3} = hlo_init   (fobj, dim_snap, Lb_snap, Ub_snap);
    states{4} = gwo_init   (fobj, dim_snap, Lb_snap, Ub_snap);
    states{5} = tow_init   (fobj, dim_snap, Lb_snap, Ub_snap);

    % ── Hub: initialise from algo bests ────────────────────
    hub_fit = min(cellfun(@(s) s.best_fit, states));
    hub_idx = find(cellfun(@(s) s.best_fit, states) == hub_fit, 1);
    hub_pos = states{hub_idx}.best_pos;

    sync_hist  = nan(N_SYNC, N_ALGO);
    conv_curve = nan(1, N_SYNC);
    accept_log = zeros(N_SYNC, N_ALGO);

    % ── Epoch loop — Concurrent Evolution Layer ────────────
    for ep = 1:N_SYNC

        % ── Step A: Each algorithm runs for NFE_EPOCH, seeded with hub_pos
        new_states = cell(1, N_ALGO);
        new_fits   = nan(1, N_ALGO);

        if use_parallel
            % Snapshot for parfor
            hub_pos_ep = hub_pos;
            hub_fit_ep = hub_fit;
            fobj_local = @(x) unified_fobj(x, DFenPei_snap, data_snap.dis, Lb_snap, Ub_snap);

            par_states  = states;
            par_results = cell(1, N_ALGO);

            parfor a = 1:N_ALGO
                st = inject_hub(par_states{a}, hub_pos_ep, hub_fit_ep);
                switch a
                    case 1, st = nsga_epoch (fobj_local, st, dim_snap, Lb_snap, Ub_snap, NFE_EPOCH, DFenPei_snap, data_snap);
                    case 2, st = abc_epoch  (fobj_local, st, dim_snap, Lb_snap, Ub_snap, NFE_EPOCH);
                    case 3, st = hlo_epoch  (fobj_local, st, dim_snap, Lb_snap, Ub_snap, NFE_EPOCH);
                    case 4, st = gwo_epoch  (fobj_local, st, dim_snap, Lb_snap, Ub_snap, NFE_EPOCH);
                    case 5, st = tow_epoch  (fobj_local, st, dim_snap, Lb_snap, Ub_snap, NFE_EPOCH);
                end
                par_results{a} = st;
            end
            new_states = par_results;
        else
            for a = 1:N_ALGO
                st = inject_hub(states{a}, hub_pos, hub_fit);
                switch a
                    case 1, st = nsga_epoch (fobj, st, dim_snap, Lb_snap, Ub_snap, NFE_EPOCH, DFenPei_snap, data_snap);
                    case 2, st = abc_epoch  (fobj, st, dim_snap, Lb_snap, Ub_snap, NFE_EPOCH);
                    case 3, st = hlo_epoch  (fobj, st, dim_snap, Lb_snap, Ub_snap, NFE_EPOCH);
                    case 4, st = gwo_epoch  (fobj, st, dim_snap, Lb_snap, Ub_snap, NFE_EPOCH);
                    case 5, st = tow_epoch  (fobj, st, dim_snap, Lb_snap, Ub_snap, NFE_EPOCH);
                end
                new_states{a} = st;
            end
        end

        % ── Step B: Collect per-algo bests, update hub ─────
        for a = 1:N_ALGO
            new_fits(a)      = new_states{a}.best_fit;
            sync_hist(ep, a) = new_fits(a);
        end

        [ep_best_fit, ep_winner] = min(new_fits);
        if ep_best_fit < hub_fit
            hub_fit = ep_best_fit;
            hub_pos = new_states{ep_winner}.best_pos;
        end

        % ── Step C: Record hub acceptance (did injection help?) ─
        for a = 1:N_ALGO
            % accepted if the injected hub was actually used (best_fit improved vs init)
            accept_log(ep, a) = double(new_fits(a) <= states{a}.best_fit);
        end

        states     = new_states;
        conv_curve(ep) = hub_fit;
    end

    % Save run results
    all_hub_fit(run_id)        = hub_fit;
    all_hub_pos{run_id}        = hub_pos;
    all_sync_hist(run_id,:,:)  = sync_hist;
    all_conv(run_id,:)         = conv_curve;
    all_accept(run_id,:,:)     = accept_log;

    [~, final_winner] = min(sync_hist(end,:));
    fprintf('%-6d | %-12.2f | %-12.2f | Final hub winner: %s\n', ...
        run_id, hub_fit, toc(t_run), ALGO_NAMES{final_winner});
end

fprintf('\n%s\n', repmat('=',1,60));
fprintf('  HYBRID SUMMARY  (%d runs)\n', N_RUNS);
fprintf('  Best  : %.2f m\n', min(all_hub_fit));
fprintf('  Mean  : %.2f m\n', mean(all_hub_fit));
fprintf('  Std   : %.2f m\n', std(all_hub_fit));
fprintf('  Worst : %.2f m\n', max(all_hub_fit));
fprintf('  Total elapsed: %.2f s\n', toc);
fprintf('%s\n\n', repmat('=',1,60));

%% ==================== 4. Visualisation ====================
sync_nfe  = round(linspace(NFE_EPOCH, NFE_TOTAL, N_SYNC));
conv_mean = mean(all_conv, 1);
conv_std  = std (all_conv, 0, 1);
sync_mean = squeeze(mean(all_sync_hist, 1));  % N_SYNC × N_ALGO

%% ── Fig 1: Convergence Curves ──────────────────────────────
fig1 = figure('Color',C_W,'Name','hybrid_fig1_convergence', ...
    'Position',[40 40 1300 560]);

ax1l = subplot(1,2,1);
hold on; box on; grid on;
ax1l.GridColor = [0.88 0.88 0.85]; ax1l.GridAlpha = 0.45; ax1l.Color = C_W;

% Individual run lines (faint) — light gold tint
for r = 1:N_RUNS
    plot(sync_nfe, all_conv(r,:), '-', 'Color', [C_GOLD 0.22], 'LineWidth', 0.8);
end
% Shaded std band — green fill
fill([sync_nfe, fliplr(sync_nfe)], ...
     [conv_mean+conv_std, fliplr(conv_mean-conv_std)], ...
     C_GREEN, 'FaceAlpha',0.18, 'EdgeColor','none');
% Mean line — deep green
plot(sync_nfe, conv_mean, '-o', 'Color', C_GREEN, 'LineWidth', 2.8, ...
     'MarkerSize', 5, 'MarkerFaceColor', C_GREEN, 'MarkerEdgeColor','w');

xlabel('NFE','FontSize',10,'FontWeight','bold');
ylabel('Hub Best Fitness (m)','FontSize',10,'FontWeight','bold');
title('Hybrid Hub — Convergence (Mean ± Std)', ...
      'FontSize',11,'FontWeight','bold','Color',[0.2 0.2 0.2]);
legend({'Individual runs','±1 Std','Hub Mean'},'Location','northeast','Box','off','FontSize',9);
set(ax1l,'TickDir','out','FontSize',9,'Box','off');

ax1r = subplot(1,2,2);
hold on; box on; grid on;
ax1r.GridColor = [0.88 0.88 0.85]; ax1r.GridAlpha = 0.45; ax1r.Color = C_W;

leg_h = gobjects(N_ALGO,1);
for a = 1:N_ALGO
    ym = sync_mean(:,a)';
    fill([sync_nfe, fliplr(sync_nfe)], ...
         [ym + squeeze(std(all_sync_hist(:,:,a),0,1))', ...
          fliplr(ym - squeeze(std(all_sync_hist(:,:,a),0,1))')], ...
         ALGO_COLORS{a}, 'FaceAlpha',0.10, 'EdgeColor','none');
    leg_h(a) = plot(sync_nfe, ym, '-', 'Color', ALGO_COLORS{a}, ...
        'LineWidth', 2.2, 'DisplayName', ALGO_NAMES{a});
end
plot(sync_nfe, conv_mean, '--', 'Color', C_GOLD, 'LineWidth', 2.0, ...
     'DisplayName', 'Hub (Global)');

xlabel('NFE','FontSize',10,'FontWeight','bold');
ylabel('Best Fitness (m)','FontSize',10,'FontWeight','bold');
title('Per-Algorithm Convergence in Hybrid', ...
      'FontSize',11,'FontWeight','bold','Color',[0.2 0.2 0.2]);
legend('Location','northeast','Box','off','FontSize',9);
set(ax1r,'TickDir','out','FontSize',9,'Box','off');

sgtitle('Parallel Hybrid — Concurrent Evolution Layer Convergence', ...
        'FontSize',13,'FontWeight','bold','Color',[0.12 0.12 0.12]);
exportgraphics(fig1,'hybrid_fig1_convergence.png','Resolution',200);
fprintf('Saved: hybrid_fig1_convergence.png\n');

%% ── Fig 2: Algo Contribution (Best-epoch wins & acceptance rate) ──
fig2 = figure('Color',C_W,'Name','hybrid_fig2_contribution', ...
    'Position',[60 60 1300 540]);

% ── 2A: Win-count per algo (which algo provided the hub best at each sync)
ax2a = subplot(1,3,1);
hold on; box on; grid on;
ax2a.GridColor = [0.88 0.88 0.85]; ax2a.GridAlpha = 0.45; ax2a.Color = C_W;

win_count = zeros(1, N_ALGO);
for r = 1:N_RUNS
    for ep = 1:N_SYNC
        ep_fits = squeeze(all_sync_hist(r, ep, :))';
        [~, w]  = min(ep_fits);
        win_count(w) = win_count(w) + 1;
    end
end

bh2a = bar(win_count, 'FaceColor','flat', 'EdgeColor','none', 'BarWidth',0.68);
for a = 1:N_ALGO, bh2a.CData(a,:) = ALGO_COLORS{a}; end
for a = 1:N_ALGO
    text(a, win_count(a) + max(win_count)*0.03, num2str(win_count(a)), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold', ...
        'Color',[0.2 0.2 0.2]);
end
xticks(1:N_ALGO); xticklabels(ALGO_NAMES); xtickangle(0);
ylabel('Epoch-Win Count','FontSize',10,'FontWeight','bold');
title('Hub Feed Wins per Algorithm','FontSize',11,'FontWeight','bold','Color',[0.2 0.2 0.2]);
set(ax2a,'TickDir','out','FontSize',9,'Box','off');

% ── 2B: Mean fitness per algo at last sync
ax2b = subplot(1,3,2);
hold on; box on; grid on;
ax2b.GridColor = [0.88 0.88 0.85]; ax2b.GridAlpha = 0.45; ax2b.Color = C_W;

final_means = mean(squeeze(all_sync_hist(:, end, :)), 1);
final_stds  = std (squeeze(all_sync_hist(:, end, :)), 0, 1);

bh2b = bar(final_means,'FaceColor','flat','EdgeColor','none','BarWidth',0.68);
for a = 1:N_ALGO, bh2b.CData(a,:) = ALGO_COLORS{a}; end
errorbar(1:N_ALGO, final_means, final_stds, ...
    '.','Color',C_GOLD*0.7,'LineWidth',1.3,'CapSize',6);
for a = 1:N_ALGO
    text(a, final_means(a)+final_stds(a)+max(final_means)*0.02, ...
        sprintf('%.0f',final_means(a)), ...
        'HorizontalAlignment','center','FontSize',9,'Color',[0.2 0.2 0.2]);
end
xticks(1:N_ALGO); xticklabels(ALGO_NAMES); xtickangle(0);
ylabel('Mean Best Fitness at Final Sync (m)','FontSize',10,'FontWeight','bold');
title('Final-Epoch Fitness per Algorithm','FontSize',11,'FontWeight','bold','Color',[0.2 0.2 0.2]);
set(ax2b,'TickDir','out','FontSize',9,'Box','off');

% ── 2C: Hub-injection acceptance rate per algo per sync (avg over runs)
ax2c = subplot(1,3,3);
hold on; box on; grid on;
ax2c.GridColor = [0.88 0.88 0.85]; ax2c.GridAlpha = 0.45; ax2c.Color = C_W;

accept_rate = squeeze(mean(all_accept, 1));   % N_SYNC × N_ALGO
for a = 1:N_ALGO
    plot(1:N_SYNC, accept_rate(:,a)*100, '-o', ...
        'Color', ALGO_COLORS{a}, 'LineWidth', 2.0, ...
        'MarkerSize', 4, 'MarkerFaceColor', ALGO_COLORS{a}, 'MarkerEdgeColor','w', ...
        'DisplayName', ALGO_NAMES{a});
end
xlabel('Sync Epoch','FontSize',10,'FontWeight','bold');
ylabel('Hub-Injection Acceptance Rate (%)','FontSize',10,'FontWeight','bold');
title('Hub Acceptance Rate over Epochs','FontSize',11,'FontWeight','bold','Color',[0.2 0.2 0.2]);
legend('Location','best','Box','off','FontSize',9);
set(ax2c,'TickDir','out','FontSize',9,'Box','off');
ylim([0 105]);

sgtitle('Parallel Hybrid — Algorithm Contribution Analysis', ...
        'FontSize',13,'FontWeight','bold','Color',[0.12 0.12 0.12]);
exportgraphics(fig2,'hybrid_fig2_contribution.png','Resolution',200);
fprintf('Saved: hybrid_fig2_contribution.png\n');

%% ── Fig 3: Sync Heatmap (fitness matrix: sync × algo, averaged over runs) ──
fig3 = figure('Color',C_W,'Name','hybrid_fig3_sync_heatmap', ...
    'Position',[80 80 1100 520]);

ax3 = axes('Color',C_W,'Position',[0.10 0.12 0.76 0.72]);
imagesc(sync_mean');       % N_ALGO × N_SYNC
colormap(ax3, cmap_gg);
cb3 = colorbar('eastoutside');
cb3.Label.String = 'Mean Best Fitness (m)';
cb3.FontSize = 8;

% Annotate cells
for a = 1:N_ALGO
    for ep = 1:N_SYNC
        text(ep, a, sprintf('%.0f', sync_mean(ep,a)), ...
            'HorizontalAlignment','center','VerticalAlignment','middle', ...
            'FontSize',7,'Color',[0.1 0.1 0.1],'FontWeight','bold');
    end
end

% Highlight the minimum per column (best algo at each sync)
for ep = 1:N_SYNC
    [~, ba] = min(sync_mean(ep,:));
    rectangle('Position',[ep-0.5, ba-0.5, 1, 1], ...
        'EdgeColor',C_GOLD,'LineWidth',2.0,'Curvature',0.12);
end

xticks(1:N_SYNC);
xticklabels(arrayfun(@(e) sprintf('S%d',e), 1:N_SYNC,'UniformOutput',false));
xtickangle(45);
yticks(1:N_ALGO); yticklabels(ALGO_NAMES);
xlabel('Sync Epoch','FontSize',10,'FontWeight','bold');
ylabel('Algorithm','FontSize',10,'FontWeight','bold');
title('Per-Algo Best Fitness at Each Sync Epoch  (box = epoch winner)', ...
    'FontSize',11,'FontWeight','bold','Color',[0.2 0.2 0.2]);

% Left colour strip
ax3s = axes('Position',[0.04 0.12 0.025 0.72],'Color','none');
for a = 1:N_ALGO
    patch(ax3s,[0 1 1 0],[a-0.5 a-0.5 a+0.5 a+0.5], ...
        ALGO_COLORS{a},'EdgeColor','none');
end
xlim(ax3s,[0 1]); ylim(ax3s,[0.5 N_ALGO+0.5]); axis(ax3s,'off');

sgtitle('Parallel Hybrid — Sync Fitness Heatmap (mean over runs)', ...
        'FontSize',13,'FontWeight','bold','Color',[0.12 0.12 0.12]);
exportgraphics(fig3,'hybrid_fig3_sync_heatmap.png','Resolution',200);
fprintf('Saved: hybrid_fig3_sync_heatmap.png\n');

fprintf('\nAll figures saved. Total time: %.2f s\n', toc);


%% ============================================================
%%  Hub Injection Helper
%%  Inserts hub_pos into the algo's population if it's better
%% ============================================================
function st = inject_hub(st, hub_pos, hub_fit)
    if hub_fit < st.best_fit
        st.best_fit = hub_fit;
        st.best_pos = hub_pos;
        % Overwrite the worst member of the population with hub solution
        if isfield(st,'pop') && ~isempty(st.pop)
            [~, worst_idx] = max(st.fit);
            st.pop(worst_idx,:) = hub_pos;
            st.fit(worst_idx)   = hub_fit;
        end
    end
end


%% ============================================================
%%  NSGA — Init & Epoch
%% ============================================================
function st = nsga_init(fobj, dim, Lb, Ub, DFenPei_, data_)
    popSize = 60;
    pop  = Lb + round(rand(popSize, dim) .* (Ub - Lb));
    fit  = arrayfun(@(i) fobj(pop(i,:)), 1:popSize)';
    [best_fit, bi] = min(fit);
    st = struct('pop',pop,'fit',fit,'best_fit',best_fit,'best_pos',pop(bi,:), ...
                'DFenPei_',{DFenPei_},'data_',data_,'popSize',popSize);
end

function st = nsga_epoch(fobj, st, dim, Lb, Ub, NFE, DFenPei_, data_)
    popSize = st.popSize;
    pop = st.pop; fit = st.fit;
    best_fit = st.best_fit; best_pos = st.best_pos;
    NFE_used = 0;
    try
        dis_mat_ = data_.dis;
    catch
        % fallback: simple GA crossover
        maxGen = floor(NFE / popSize);
        for g = 1:maxGen
            child = zeros(popSize, dim);
            for i = 1:popSize
                p1 = pop(randi(popSize),:); p2 = pop(randi(popSize),:);
                cp = randi(dim);
                child(i,:) = [p1(1:cp-1), p2(cp:end)];
                if rand < 0.1
                    d_ = randi(dim);
                    child(i,d_) = Lb(d_) + randi(max(1,Ub(d_)-Lb(d_)+1)) - 1;
                end
                child(i,:) = max(Lb, min(Ub, child(i,:)));
            end
            for i = 1:popSize
                f = fobj(child(i,:)); NFE_used = NFE_used + 1;
                if f < fit(i), pop(i,:) = child(i,:); fit(i) = f; end
                if f < best_fit, best_fit = f; best_pos = child(i,:); end
                if NFE_used >= NFE, break; end
            end
            if NFE_used >= NFE, break; end
        end
        st.pop = pop; st.fit = fit; st.best_fit = best_fit; st.best_pos = best_pos;
        return;
    end
    P_ = cellfun(@(x) length(x)-1, DFenPei_);
    nsga_obj = @(x) nsga_eval_local(x, DFenPei_, dis_mat_, P_);
    maxGen = floor(NFE / popSize);
    F = zeros(popSize,2);
    for i = 1:popSize, F(i,:) = nsga_obj(pop(i,:)); end
    for g = 1:maxGen
        child = zeros(popSize, dim);
        for i = 1:popSize
            p1 = pop(randi(popSize),:); p2 = pop(randi(popSize),:);
            cp = randi(dim);
            child(i,:) = [p1(1:cp-1), p2(cp:end)];
            if rand < 0.1
                d_ = randi(dim);
                child(i,d_) = Lb(d_) + randi(max(1,Ub(d_)-Lb(d_)+1)) - 1;
            end
            child(i,:) = max(Lb, min(Ub, child(i,:)));
        end
        Fc = zeros(popSize,2);
        for i = 1:popSize
            Fc(i,:) = nsga_obj(child(i,:)); NFE_used = NFE_used + 1;
        end
        combined = [pop; child]; Fall = [F; Fc];
        [~, si] = sort(Fall(:,1));
        pop = combined(si(1:popSize),:); F = Fall(si(1:popSize),:);
        [cur_best, bi] = min(F(:,1));
        real_fit = fobj(pop(bi,:));
        if real_fit < best_fit, best_fit = real_fit; best_pos = pop(bi,:); end
        if NFE_used >= NFE, break; end
    end
    fit = arrayfun(@(i) fobj(pop(i,:)), 1:popSize)';
    st.pop = pop; st.fit = fit; st.best_fit = best_fit; st.best_pos = best_pos;
end

function f2 = nsga_eval_local(x, DFenPei_, dis_mat_, P_)
    X = max(1, min(round(x), P_)); total_d = 0; Y = zeros(1, size(dis_mat_,2));
    for i = 1:length(X)
        hid = DFenPei_{i}(1); eid = DFenPei_{i}(X(i)+1);
        total_d = total_d + dis_mat_(hid,eid); Y(eid) = Y(eid) + 12;
    end
    f2 = [total_d, var(Y)];
end


%% ============================================================
%%  ABC — Init & Epoch
%% ============================================================
function st = abc_init(fobj, dim, Lb, Ub)
    FN = 30;
    pop   = Lb + round(rand(FN, dim) .* (Ub - Lb));
    fit   = arrayfun(@(i) fobj(pop(i,:)), 1:FN)';
    trial = zeros(1, FN);
    [best_fit, bi] = min(fit);
    st = struct('pop',pop,'fit',fit,'trial',trial,'best_fit',best_fit, ...
                'best_pos',pop(bi,:),'FN',FN,'limit',25);
end

function st = abc_epoch(fobj, st, dim, Lb, Ub, NFE)
    FN = st.FN; pop = st.pop; fit = st.fit; trial = st.trial;
    best_fit = st.best_fit; best_pos = st.best_pos;
    limit = st.limit;
    NFE_used = 0;
    maxCycle = floor(NFE / (FN*2));
    for iter_ = 1:maxCycle
        % Employed bees
        for i = 1:FN
            k = i; while k == i, k = randi(FN); end
            phi = randi([-1,1], 1, dim);
            nf = max(Lb, min(Ub, round(pop(i,:) + phi .* (pop(i,:) - pop(k,:)))));
            fn = fobj(nf); NFE_used = NFE_used + 1;
            if fn < fit(i), pop(i,:) = nf; fit(i) = fn; trial(i) = 0;
            else, trial(i) = trial(i) + 1; end
            if fn < best_fit, best_fit = fn; best_pos = nf; end
        end
        % Onlooker bees (probability selection)
        prob = (1 ./ (fit + 1e-10)) / sum(1 ./ (fit + 1e-10));
        for b = 1:FN
            i = find(rand <= cumsum(prob), 1);
            if isempty(i), i = randi(FN); end
            k = i; while k == i, k = randi(FN); end
            phi = randi([-1,1], 1, dim);
            nf = max(Lb, min(Ub, round(pop(i,:) + phi .* (pop(i,:) - pop(k,:)))));
            fn = fobj(nf); NFE_used = NFE_used + 1;
            if fn < fit(i), pop(i,:) = nf; fit(i) = fn; trial(i) = 0;
            else, trial(i) = trial(i) + 1; end
            if fn < best_fit, best_fit = fn; best_pos = nf; end
        end
        % Scout bees
        for i = 1:FN
            if trial(i) > limit
                pop(i,:) = Lb + round(rand(1,dim) .* (Ub - Lb));
                fit(i)   = fobj(pop(i,:)); trial(i) = 0; NFE_used = NFE_used + 1;
                if fit(i) < best_fit, best_fit = fit(i); best_pos = pop(i,:); end
            end
        end
        if NFE_used >= NFE, break; end
    end
    st.pop = pop; st.fit = fit; st.trial = trial;
    st.best_fit = best_fit; st.best_pos = best_pos;
end


%% ============================================================
%%  HLO — Init & Epoch  (Human Learning Optimisation)
%% ============================================================
function st = hlo_init(fobj, dim, Lb, Ub)
    popSize = 40; bpv = 8; m = dim * bpv;
    bin_pop = randi([0,1], popSize, m);
    decode  = @(row) arrayfun(@(v) max(Lb(v), min(Ub(v), Lb(v) + round( ...
        sum(row((v-1)*bpv+1:v*bpv) .* (2.^(bpv-1:-1:0))) / (2^bpv-1) * (Ub(v)-Lb(v))))), 1:dim);
    IKD    = bin_pop;
    IKDfit = arrayfun(@(i) fobj(decode(bin_pop(i,:))), 1:popSize)';
    [best_val, bi] = min(IKDfit);
    SKD    = IKD(bi,:); SKDfit = best_val;
    best_pos = decode(SKD);
    pop_int = zeros(popSize, dim);
    for i = 1:popSize, pop_int(i,:) = decode(bin_pop(i,:)); end
    st = struct('bin_pop',bin_pop,'IKD',IKD,'IKDfit',IKDfit, ...
                'SKD',SKD,'SKDfit',SKDfit,'decode',decode, ...
                'pop',pop_int,'fit',IKDfit, ...
                'best_fit',best_val,'best_pos',best_pos, ...
                'popSize',popSize,'bpv',bpv,'m',m,'p_r',0.1,'p_i',0.5);
end

function st = hlo_epoch(fobj, st, dim, Lb, Ub, NFE)
    popSize = st.popSize; m = st.m;
    bin_pop = st.bin_pop; IKD = st.IKD; IKDfit = st.IKDfit;
    SKD = st.SKD; SKDfit = st.SKDfit;
    best_fit = st.best_fit; best_pos = st.best_pos;
    decode   = st.decode;
    p_r = st.p_r; p_i = st.p_i;
    NFE_used = 0;
    maxIter  = floor(NFE / popSize);
    for iter_ = 1:maxIter
        for i = 1:popSize
            for j = 1:m
                pr = rand;
                if     pr < p_r,       bin_pop(i,j) = randi([0,1]);
                elseif pr < p_i,       bin_pop(i,j) = IKD(i,j);
                else,                  bin_pop(i,j) = SKD(j); end
            end
            x_int = decode(bin_pop(i,:)); fv = fobj(x_int); NFE_used = NFE_used + 1;
            if fv < IKDfit(i), IKDfit(i) = fv; IKD(i,:) = bin_pop(i,:); end
            if fv < SKDfit,    SKDfit = fv; SKD = bin_pop(i,:); best_pos = x_int; best_fit = fv; end
        end
        if NFE_used >= NFE, break; end
    end
    pop_int = zeros(popSize, dim);
    for i = 1:popSize, pop_int(i,:) = decode(bin_pop(i,:)); end
    st.bin_pop = bin_pop; st.IKD = IKD; st.IKDfit = IKDfit;
    st.SKD = SKD; st.SKDfit = SKDfit;
    st.pop = pop_int; st.fit = IKDfit;
    st.best_fit = best_fit; st.best_pos = best_pos;
end


%% ============================================================
%%  GWO — Init & Epoch  (Grey Wolf Optimiser)
%% ============================================================
function st = gwo_init(fobj, dim, Lb, Ub)
    n   = 50;
    pop = Lb + round(rand(n, dim) .* (Ub - Lb));
    fit = arrayfun(@(i) fobj(pop(i,:)), 1:n)';
    [~, si] = sort(fit);
    st = struct('pop',pop,'fit',fit,'n',n, ...
                'Ap',pop(si(1),:),'As',fit(si(1)), ...
                'Bp',pop(si(2),:),'Bs',fit(si(2)), ...
                'Dp',pop(si(3),:),'Ds',fit(si(3)), ...
                'best_fit',fit(si(1)),'best_pos',pop(si(1),:), ...
                'iter_total',0,'maxIter',1000);
end

function st = gwo_epoch(fobj, st, dim, Lb, Ub, NFE)
    n = st.n; pop = st.pop; fit = st.fit;
    Ap = st.Ap; As = st.As;
    Bp = st.Bp; Bs = st.Bs;
    Dp = st.Dp; Ds = st.Ds;
    best_fit = st.best_fit; best_pos = st.best_pos;
    iter_total = st.iter_total;
    maxIter = max(st.maxIter, floor(NFE/n));
    NFE_used = 0;
    for l = 1:floor(NFE/n)
        iter_total = iter_total + 1;
        a = 2 - iter_total * (2 / maxIter);
        for i = 1:n
            np = zeros(1, dim);
            for j = 1:dim
                X1 = Ap(j) - (2*a*rand-a) * abs(2*rand*Ap(j) - pop(i,j));
                X2 = Bp(j) - (2*a*rand-a) * abs(2*rand*Bp(j) - pop(i,j));
                X3 = Dp(j) - (2*a*rand-a) * abs(2*rand*Dp(j) - pop(i,j));
                np(j) = round((X1+X2+X3)/3);
            end
            np = max(Lb, min(Ub, np)); f = fobj(np); NFE_used = NFE_used + 1;
            pop(i,:) = np; fit(i) = f;
            if f < As, Ds=Bs;Dp=Bp;Bs=As;Bp=Ap;As=f;Ap=np;
            elseif f < Bs, Ds=Bs;Dp=Bp;Bs=f;Bp=np;
            elseif f < Ds, Ds=f;Dp=np; end
        end
        if As < best_fit, best_fit = As; best_pos = Ap; end
        if NFE_used >= NFE, break; end
    end
    st.pop = pop; st.fit = fit;
    st.Ap = Ap; st.As = As; st.Bp = Bp; st.Bs = Bs; st.Dp = Dp; st.Ds = Ds;
    st.best_fit = best_fit; st.best_pos = best_pos; st.iter_total = iter_total;
end


%% ============================================================
%%  TOW — Init & Epoch  (Tug-of-War Optimisation)
%% ============================================================
function st = tow_init(fobj, dim, Lb, Ub)
    nT   = 35;
    pop  = double(Lb + round(rand(nT, dim) .* (Ub - Lb)));
    fit  = arrayfun(@(i) fobj(pop(i,:)), 1:nT)';
    [best_fit, bi] = min(fit);
    st = struct('pop',pop,'fit',fit,'nT',nT, ...
                'best_fit',best_fit,'best_pos',pop(bi,:), ...
                'alpha_t',0.97,'sigma0',2.5,'it_total',0);
end

function st = tow_epoch(fobj, st, dim, Lb, Ub, NFE)
    nT = st.nT; pop = st.pop; fit = st.fit;
    alpha_t = st.alpha_t; sigma0 = st.sigma0;
    best_fit = st.best_fit; best_pos = st.best_pos;
    it_total = st.it_total;
    NFE_used = 0;
    for it = 1:floor(NFE/nT)
        it_total = it_total + 1;
        inv_f = 1 ./ (fit - min(fit) + 1e-10);
        W = inv_f / sum(inv_f);
        wc = W' * pop;
        sigma = sigma0 * alpha_t^it_total;
        for i = 1:nT
            step = round(0.6*(wc - pop(i,:)) + sigma*randn(1,dim));
            np   = max(Lb, min(Ub, pop(i,:) + step));
            f    = fobj(np); NFE_used = NFE_used + 1;
            if f < fit(i), pop(i,:) = np; fit(i) = f; end
            if f < best_fit, best_fit = f; best_pos = np; end
        end
        if NFE_used >= NFE, break; end
    end
    st.pop = pop; st.fit = fit;
    st.best_fit = best_fit; st.best_pos = best_pos; st.it_total = it_total;
end


%% ============================================================
%%  Objective Function
%% ============================================================
function fitness = unified_fobj(x, DFenPei, dis_mat, Lb, Ub)
    fitness = 0;
    for i = 1:length(DFenPei)
        idx     = max(Lb(i), min(round(x(i)), Ub(i)));
        fitness = fitness + dis_mat(DFenPei{i}(1), DFenPei{i}(idx+1));
    end
end