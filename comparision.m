%% ============================================================
%  Pipeline vs Parallel — 联合对比可视化脚本
%
%  使用前提：先在同一个 MATLAB 工作空间中依次运行：
%    1. PIPE_20_55.m         （运行完后不要 clear）
%    2. 本脚本直接接着运行
%
%  本脚本会：
%    - 保存 Pipeline 的结果到临时变量
%    - 运行 Parallel 并保存其结果
%    - 图A：两条收敛曲线画在同一张图（Pipeline分5段分色 + Parallel单色）
%    - 图B：合并分配地图（Pipeline红色底 + Parallel差异线深绿色覆盖）
%% ============================================================

fprintf('\n[对比脚本] 保存 Pipeline 结果...\n');

%% ===== Step 1: 保存 Pipeline 结果（假设工作空间已有）=====
pipe_X_final      = X_final;          % Pipeline 最终整数解
pipe_CC_all       = CC_all;           % Pipeline 完整收敛序列
pipe_stage_boundary = stage_boundary; % Pipeline 阶段边界
pipe_stage_starts   = stage_starts;   % Pipeline 阶段起点
pipe_stage_names    = stage_names;    % Pipeline 各阶段算法名
pipe_stage_colors   = stage_colors;   % Pipeline 各阶段颜色
pipe_bestFit        = bestFit5;       % Pipeline 最终适应度

% 保留公共数据（两者共用）
pipe_house_x  = house_x;
pipe_house_y  = house_y;
pipe_binan_x  = binan_x;
pipe_binan_y  = binan_y;
pipe_offset_x = offset_x;
pipe_offset_y = offset_y;
pipe_dim      = dim;
pipe_Lb       = Lb;
pipe_Ub       = Ub;
pipe_FID      = FID;
pipe_DFenPei  = DFenPei;

%% ===== Step 2: 运行 Parallel（内联核心逻辑，避免 clear 覆盖）=====
fprintf('[对比脚本] 运行 Parallel Integrative...\n');

% 重新加载数据确保 Parallel 用同一份数据
if exist('sj5.mat','file'), load('sj5.mat'); end
if exist('dis','var'), data.dis = dis; end

n_DE=30; n_GWO=30; n_CS=20; n_TOW=20; n_PO=60;
N_total = n_DE+n_GWO+n_CS+n_TOW+n_PO;
MaxSyncIter = floor(150000/N_total);
K_elite=20; sync_freq=5;
F_DE=0.7; CR_DE=0.8; beta_CS=1.5;
alpha_tow=0.995; sigma_init=4.0;
n_parties=3; n_members=20; a_init=2.0;

par_fobj_h = @(x) par_fobj_cmp(x, DFenPei, data.dis, pipe_Lb, pipe_Ub);

init_fn = @(n,d,L,U) round(repmat(L,n,1)+rand(n,d).*repmat(U-L,n,1));
pop_DE  = init_fn(n_DE,  pipe_dim, pipe_Lb, pipe_Ub);
pop_GWO = init_fn(n_GWO, pipe_dim, pipe_Lb, pipe_Ub);
pop_CS  = init_fn(n_CS,  pipe_dim, pipe_Lb, pipe_Ub);
pop_TOW = init_fn(n_TOW, pipe_dim, pipe_Lb, pipe_Ub);
pop_PO  = init_fn(n_PO,  pipe_dim, pipe_Lb, pipe_Ub);

fit_DE  = arrayfun(@(i) par_fobj_h(pop_DE(i,:)),  1:n_DE)';
fit_GWO = arrayfun(@(i) par_fobj_h(pop_GWO(i,:)), 1:n_GWO)';
fit_CS  = arrayfun(@(i) par_fobj_h(pop_CS(i,:)),  1:n_CS)';
fit_TOW = arrayfun(@(i) par_fobj_h(pop_TOW(i,:)), 1:n_TOW)';
fit_PO  = arrayfun(@(i) par_fobj_h(pop_PO(i,:)),  1:n_PO)';

all_pop0=[pop_DE;pop_GWO;pop_CS;pop_TOW;pop_PO];
all_fit0=[fit_DE;fit_GWO;fit_CS;fit_TOW;fit_PO];
[~,si0]=sort(all_fit0);
ep0=all_pop0(si0(1:K_elite),:); ef0=all_fit0(si0(1:K_elite));
global_best_pos=ep0(1,:); global_best_fit=ef0(1);

[~,gs]=sort(fit_GWO);
Alpha_pos=pop_GWO(gs(1),:); Alpha_score=fit_GWO(gs(1));
Beta_pos =pop_GWO(gs(2),:); Beta_score =fit_GWO(gs(2));
Delta_pos=pop_GWO(gs(3),:); Delta_score=fit_GWO(gs(3));

CC_global=zeros(MaxSyncIter,1);

for iter=1:MaxSyncIter
    for i=1:n_DE
        r=randperm(n_DE,3);
        v=round(pop_DE(i,:)+F_DE*(global_best_pos-pop_DE(i,:))+F_DE*(pop_DE(r(1),:)-pop_DE(r(2),:)));
        v=max(pipe_Lb,min(pipe_Ub,v));
        mask=rand(1,pipe_dim)<CR_DE; mask(randi(pipe_dim))=true;
        trial=pop_DE(i,:); trial(mask)=v(mask);
        ft=par_fobj_h(trial);
        if ft<fit_DE(i), pop_DE(i,:)=trial; fit_DE(i)=ft; end
    end
    [best_DE_fit,bi]=min(fit_DE); best_DE_pos=pop_DE(bi,:);

    a_gwo=a_init-iter*(a_init/MaxSyncIter);
    for i=1:n_GWO
        for j=1:pipe_dim
            X1=Alpha_pos(j)-(2*a_gwo*rand()-a_gwo)*abs(2*rand()*Alpha_pos(j)-pop_GWO(i,j));
            X2=Beta_pos(j) -(2*a_gwo*rand()-a_gwo)*abs(2*rand()*Beta_pos(j) -pop_GWO(i,j));
            X3=Delta_pos(j)-(2*a_gwo*rand()-a_gwo)*abs(2*rand()*Delta_pos(j)-pop_GWO(i,j));
            pop_GWO(i,j)=round((X1+X2+X3)/3);
        end
        pop_GWO(i,:)=max(pipe_Lb,min(pipe_Ub,pop_GWO(i,:)));
        fit_GWO(i)=par_fobj_h(pop_GWO(i,:));
        if fit_GWO(i)<Alpha_score
            Delta_score=Beta_score;Delta_pos=Beta_pos;
            Beta_score=Alpha_score;Beta_pos=Alpha_pos;
            Alpha_score=fit_GWO(i);Alpha_pos=pop_GWO(i,:);
        end
    end

    for i=1:n_CS
        levy=par_levy_cmp(beta_CS,pipe_dim);
        nn=round(pop_CS(i,:)+0.01*levy.*(pop_CS(i,:)-global_best_pos));
        nn=max(pipe_Lb,min(pipe_Ub,nn));
        ft=par_fobj_h(nn);
        if ft<fit_CS(i), pop_CS(i,:)=nn; fit_CS(i)=ft; end
    end
    [best_CS_fit,bi]=min(fit_CS); best_CS_pos=pop_CS(bi,:);

    sigma_cur=sigma_init*alpha_tow^iter;
    for i=1:n_TOW
        step=round(0.5*(global_best_pos-pop_TOW(i,:))+sigma_cur*randn(1,pipe_dim));
        pop_TOW(i,:)=max(pipe_Lb,min(pipe_Ub,pop_TOW(i,:)+step));
        fit_TOW(i)=par_fobj_h(pop_TOW(i,:));
    end
    [best_TOW_fit,bi]=min(fit_TOW); best_TOW_pos=pop_TOW(bi,:);

    for p=1:n_parties
        idx_p=(p-1)*n_members+1:p*n_members;
        [~,ll]=min(fit_PO(idx_p)); ldr_pos=pop_PO(idx_p(ll),:);
        for m=idx_p
            pop_PO(m,:)=max(pipe_Lb,min(pipe_Ub,round(pop_PO(m,:)+rand()*(ldr_pos-pop_PO(m,:)))));
            fit_PO(m)=par_fobj_h(pop_PO(m,:));
        end
    end
    [best_PO_fit,bi]=min(fit_PO); best_PO_pos=pop_PO(bi,:);

    sub_fits=[best_DE_fit,Alpha_score,best_CS_fit,best_TOW_fit,best_PO_fit];
    sub_pos =[best_DE_pos;Alpha_pos;best_CS_pos;best_TOW_pos;best_PO_pos];
    [cur_best,cb_idx]=min(sub_fits);
    if cur_best<global_best_fit
        global_best_fit=cur_best; global_best_pos=sub_pos(cb_idx,:);
    end

    if mod(iter,sync_freq)==0
        all_pc=[pop_DE;pop_GWO;pop_CS;pop_TOW;pop_PO];
        all_fc=[fit_DE;fit_GWO;fit_CS;fit_TOW;fit_PO];
        [~,si_c]=sort(all_fc);
        ep_c=all_pc(si_c(1:K_elite),:);
        [~,w]=max(fit_DE);  pop_DE(w,:) =ep_c(randi(K_elite),:); fit_DE(w) =par_fobj_h(pop_DE(w,:));
        [~,w]=max(fit_GWO); pop_GWO(w,:)=ep_c(randi(K_elite),:); fit_GWO(w)=par_fobj_h(pop_GWO(w,:));
    end
    CC_global(iter)=global_best_fit;
    if mod(iter,100)==0
        fprintf('  Parallel Iter %4d/%d | Best: %.2f m\n',iter,MaxSyncIter,global_best_fit);
    end
end

par_X_final  = global_best_pos;
par_CC       = CC_global;
par_bestFit  = global_best_fit;
fprintf('[对比脚本] Parallel 完成 | 最优: %.2f m\n\n', par_bestFit);

%% ===== Step 3: 图A — 收敛曲线对比（两者画在同一张图）=====
fprintf('[对比脚本] 绘制收敛曲线对比图...\n');

figure('Name','Pipeline vs Parallel 收敛对比','Color','w','Position',[50,50,1050,500]);
hold on; box on; grid on;

% --- Pipeline：5段分色 ---
for s=1:5
    sx = (pipe_stage_starts(s):pipe_stage_boundary(s))';
    sy = pipe_CC_all(pipe_stage_starts(s):pipe_stage_boundary(s));
    plot(sx, sy, 'Color', pipe_stage_colors{s}, 'LineWidth', 2.2, ...
         'DisplayName', ['Pipeline S' num2str(s) ': ' pipe_stage_names{s}]);
    if s<5
        xline(pipe_stage_boundary(s), '--', 'Color',[0.65 0.65 0.65], ...
              'LineWidth',0.9,'HandleVisibility','off');
        text(pipe_stage_boundary(s)+3, pipe_CC_all(pipe_stage_boundary(s)), ...
             ['→' pipe_stage_names{s+1}], 'FontSize',7.5, ...
             'Color',[0.35 0.35 0.35],'HandleVisibility','off');
    end
end

% --- Parallel：单色虚线，X轴映射到同等长度 ---
% Pipeline 总步数 vs Parallel 总步数可能不同，做 X 轴等比映射使终点对齐
pipe_len = length(pipe_CC_all);
par_len  = MaxSyncIter;
par_x    = linspace(1, pipe_len, par_len);   % 等比拉伸到同一X轴范围

plot(par_x, par_CC, 'Color',[0.10 0.10 0.10], 'LineWidth',2.0, ...
     'LineStyle','--', 'DisplayName','Parallel: DE‖GWO‖CS‖TOW‖PO');

% 标注两者最终值
ymin_pipe = pipe_CC_all(end);
ymin_par  = par_CC(end);
plot(pipe_len, ymin_pipe, 'o', 'MarkerSize',8, 'MarkerFaceColor', pipe_stage_colors{5}, ...
     'MarkerEdgeColor','k','HandleVisibility','off');
plot(pipe_len, ymin_par,  's', 'MarkerSize',8, 'MarkerFaceColor',[0.1 0.1 0.1], ...
     'MarkerEdgeColor','k','HandleVisibility','off');
text(pipe_len-length(pipe_CC_all)*0.04, ymin_pipe*1.0005, ...
     sprintf('Pipeline: %.0f m', ymin_pipe+0), ...  % +alldis_fixed 若需要可加
     'FontSize',9,'Color',pipe_stage_colors{5},'FontWeight','bold','HorizontalAlignment','right');
text(pipe_len-length(pipe_CC_all)*0.04, ymin_par*0.9995, ...
     sprintf('Parallel: %.0f m', ymin_par+0), ...
     'FontSize',9,'Color',[0.1 0.1 0.1],'FontWeight','bold','HorizontalAlignment','right');

xlabel('Cumulative Iteration (NFE-normalized)', 'FontWeight','bold');
ylabel('Best Fitness — Total Distance (m)', 'FontWeight','bold');
title('Pipeline vs Parallel Integrative — Convergence Comparison', 'FontSize',13,'FontWeight','bold');
legend('Location','northeast','FontSize',9);

%% ===== Step 4: 图B — 合并分配地图 =====
fprintf('[对比脚本] 绘制合并分配地图...\n');

% 预先计算每个 DFenPei{i} 点的目标避难所索引
pipe_eIdx = zeros(1, pipe_dim);
par_eIdx  = zeros(1, pipe_dim);
for i=1:pipe_dim
    oi_pipe = max(1, min(pipe_X_final(i), length(pipe_DFenPei{i})-1));
    pipe_eIdx(i) = pipe_DFenPei{i}(oi_pipe+1);

    oi_par  = max(1, min(par_X_final(i),  length(pipe_DFenPei{i})-1));
    par_eIdx(i)  = pipe_DFenPei{i}(oi_par+1);
end

% 哪些点分配目标不同
diff_mask = (pipe_eIdx ~= par_eIdx);   % logical 向量，true=两者分配不同
n_diff = sum(diff_mask);
fprintf('  分配差异点数: %d / %d (%.1f%%)\n', n_diff, pipe_dim, 100*n_diff/pipe_dim);

figure('Color','w','Name','Pipeline vs Parallel 分配对比地图','Position',[80,80,950,840]);
hold on; box on;

% 1. 道路底图
if isfield(data,'road')
    for i=1:length(data.road)
        plot(data.road{i}(:,1)-pipe_offset_x, data.road{i}(:,2)-pipe_offset_y, ...
             'Color',[0.88 0.88 0.88],'LineWidth',0.5);
    end
end

% 2. Pipeline 分配线（全部，红色半透明作为底层）
for i=1:pipe_dim
    oi = max(1, min(pipe_X_final(i), length(pipe_DFenPei{i})-1));
    line([pipe_house_x(pipe_DFenPei{i}(1)),   pipe_binan_x(pipe_DFenPei{i}(oi+1))], ...
         [pipe_house_y(pipe_DFenPei{i}(1)),   pipe_binan_y(pipe_DFenPei{i}(oi+1))], ...
         'Color',[1 0 0 0.10],'LineWidth',0.35);
end
% 固定分配线（也画红色）
for i=1:size(pipe_FID,1)
    line([pipe_house_x(pipe_FID(i,1)), pipe_binan_x(pipe_FID(i,2))], ...
         [pipe_house_y(pipe_FID(i,1)), pipe_binan_y(pipe_FID(i,2))], ...
         'Color',[1 0 0 0.10],'LineWidth',0.35);
end

% 3. Parallel 差异分配线（深绿色，只画和Pipeline不同的点）
h_diff_drawn = false;
for i=1:pipe_dim
    if diff_mask(i)
        oi_par = max(1, min(par_X_final(i), length(pipe_DFenPei{i})-1));
        if ~h_diff_drawn
            h_diff = line([pipe_house_x(pipe_DFenPei{i}(1)), pipe_binan_x(pipe_DFenPei{i}(oi_par+1))], ...
                          [pipe_house_y(pipe_DFenPei{i}(1)), pipe_binan_y(pipe_DFenPei{i}(oi_par+1))], ...
                          'Color',[0.0 0.50 0.15 0.55],'LineWidth',0.7);
            h_diff_drawn = true;
        else
            line([pipe_house_x(pipe_DFenPei{i}(1)), pipe_binan_x(pipe_DFenPei{i}(oi_par+1))], ...
                 [pipe_house_y(pipe_DFenPei{i}(1)), pipe_binan_y(pipe_DFenPei{i}(oi_par+1))], ...
                 'Color',[0.0 0.50 0.15 0.55],'LineWidth',0.7,'HandleVisibility','off');
        end
    end
end

% 4. 节点（最上层）
h_res = scatter(pipe_house_x, pipe_house_y, 8, [0 0.2 0.6], 'filled','MarkerFaceAlpha',0.5);
h_shl = scatter(pipe_binan_x, pipe_binan_y, 70, 'g','^','filled','MarkerEdgeColor','k');

% 5. 差异点的住宅用橙色小点高亮
if any(diff_mask)
    diff_idx = find(diff_mask);
    diff_house_ids = arrayfun(@(i) pipe_DFenPei{i}(1), diff_idx);
    scatter(pipe_house_x(diff_house_ids), pipe_house_y(diff_house_ids), ...
            18, [0.95 0.45 0.0], 'filled', 'HandleVisibility','off');
end

axis equal; axis tight; grid on;
ax=gca; ax.XAxis.Exponent=0; ax.YAxis.Exponent=0;
title(sprintf('Pipeline vs Parallel — Allocation Comparison  (差异点: %d/%d = %.1f%%)', ...
      n_diff, pipe_dim, 100*n_diff/pipe_dim), 'FontSize',12,'FontWeight','bold');

% 图例
if h_diff_drawn
    h_pipe_line = line(nan,nan,'Color',[1 0 0 0.5],'LineWidth',1.5);
    legend([h_shl, h_res, h_pipe_line, h_diff], ...
           {'Shelter','Residential','Pipeline 分配（红）','Parallel 差异线（深绿）'}, ...
           'Location','northeast','FontSize',10);
else
    h_pipe_line = line(nan,nan,'Color',[1 0 0 0.5],'LineWidth',1.5);
    legend([h_shl, h_res, h_pipe_line], ...
           {'Shelter','Residential','Pipeline 分配（红，两者完全一致）'}, ...
           'Location','northeast','FontSize',10);
    fprintf('  注意：两者分配完全一致，无差异线。\n');
end

fprintf('\n[完成] 对比可视化生成结束。\n');

%% ===== 子函数 =====
function fitness = par_fobj_cmp(x, DFenPei, dis_mat, Lb, Ub)
    fitness=0;
    for i=1:length(DFenPei)
        idx=max(Lb(i), min(round(x(i)), Ub(i)));
        fitness=fitness+dis_mat(DFenPei{i}(1), DFenPei{i}(idx+1));
    end
end

function levy = par_levy_cmp(beta, dim)
    sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    levy=randn(1,dim)*sigma./(abs(randn(1,dim)).^(1/beta));
end