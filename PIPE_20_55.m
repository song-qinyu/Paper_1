%% ============================================================
%  Pipeline Integrative Hybrid — 数据驱动最优组合正式版
%
%  竞选结果确定的最优五阶段组合：
%    Stage 1 (探索)   → ABC   (人工蜂群)      — 全局多样性探索
%    Stage 2 (收缩)   → NSGA  (非支配排序GA)  — 多目标引导精化
%    Stage 3 (脱困)   → NSGA  (非支配排序GA)  — 跳出局部最优
%    Stage 4 (精化)   → NSGA  (非支配排序GA)  — 深度精化
%    Stage 5 (深收敛) → TOW   (拔河优化)      — 最终深度收敛
%
%  精英池传递机制：每阶段输出 Top-K 精英解传入下一阶段
%  统一目标函数：整数编码总疏散距离（unified_fobj）
%  输出：8项评价指标 + 收敛曲线 + 阶段贡献图 + 2D分配地图
%% ============================================================
clc; clear; close all; tic;

fprintf('========================================================\n');
fprintf('  Pipeline Integrative Hybrid\n');
fprintf('  Stage1:ABC → Stage2:NSGA → Stage3:NSGA\n');
fprintf('         → Stage4:NSGA → Stage5:TOW\n');
fprintf('========================================================\n\n');

%% ======================== 0. 数据加载 ========================
if exist('sj5.mat','file'), load('sj5.mat');
else, error('未找到 sj5.mat'); end

if exist('dis','var'), data.dis = dis; end

raw_x = data.start(:,1); raw_y = data.start(:,2);
offset_x = min(raw_x); offset_y = min(raw_y);
house_x = raw_x - offset_x;   house_y = raw_y - offset_y;
binan_x = data.binan(:,1) - offset_x;
binan_y = data.binan(:,2) - offset_y;

dim = length(DFenPei);
Lb  = ones(1, dim);
Ub  = arrayfun(@(i) length(DFenPei{i})-1, 1:dim);

% 固定分配
FID=[]; alldis_fixed=0; YFP=zeros(1,size(data.binan,1));
for k=1:length(B)
    if length(B{k})==1
        tb=B{k}; FID=[FID;k,tb];
        alldis_fixed=alldis_fixed+data.dis(k,tb);
        YFP(tb)=YFP(tb)+12;
    end
end

% 统一目标函数（整数编码，总疏散距离）
fobj = @(x) unified_fobj(x, DFenPei, data.dis, Lb, Ub);

%% ======================== 参数设置 ========================
% 精英池大小（阶段间传递）
K_elite = 30;

% 各阶段预算（函数评价次数，向后期倾斜）
budget = struct();
budget.S1_ABC  = 20000;   % Stage1 全局探索
budget.S2_NSGA = 25000;   % Stage2 多目标收缩
budget.S3_NSGA = 25000;   % Stage3 脱困
budget.S4_NSGA = 30000;   % Stage4 精化
budget.S5_TOW  = 50000;   % Stage5 深收敛（最多预算）

% 用于颜色和名称
stage_names  = {'ABC','NSGA','NSGA','NSGA','TOW'};
stage_colors = {[0.10 0.70 0.30], [0.20 0.50 0.90], ...
                [0.90 0.50 0.10], [0.75 0.10 0.75], [0.80 0.15 0.15]};

CC_all = []; stage_boundary = zeros(1,5);

%% ============================================================
%  STAGE 1: ABC — 全局多样性探索
%  无精英输入，全随机初始化，利用ABC的觅食多样性覆盖解空间
%% ============================================================
fprintf('[Stage 1] ABC  — 全局探索...\n');

FN1    = 40;                              % 食物源数量
limit1 = 25;                              % 废弃阈值
maxCyc1= floor(budget.S1_ABC / (FN1*2)); % 每轮评价 FN*2 次

% 全随机初始化（Stage1不依赖精英）
pop1 = zeros(FN1, dim);
for i=1:FN1
    pop1(i,:) = Lb + round(rand(1,dim).*(Ub-Lb));
end
fit1  = arrayfun(@(i) fobj(pop1(i,:)), 1:FN1)';
trial1= zeros(1,FN1);
[bestFit1, bi] = min(fit1);
bestPos1 = pop1(bi,:);
CC1 = zeros(maxCyc1,1);

for iter=1:maxCyc1
    % 雇佣蜂阶段
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
    % 侦查蜂（废弃并随机重生）
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

% 输出精英池
[~,si]=sort(fit1);
elite_pop = pop1(si(1:min(K_elite,FN1)),:);
elite_fit = fit1(si(1:min(K_elite,FN1)));

CC_all=[CC_all;CC1]; stage_boundary(1)=length(CC_all);
fprintf('  ABC完成 | 最优: %.2f m | 精英池: %d 个解\n\n', bestFit1, size(elite_pop,1));

%% ============================================================
%  STAGE 2: NSGA — 多目标引导收缩
%  接收Stage1精英池，以Pareto前沿最优解引导种群向好区域聚拢
%% ============================================================
fprintf('[Stage 2] NSGA — 精英引导多目标收缩...\n');

popSz2 = 60;
maxGen2= floor(budget.S2_NSGA / popSz2);
P_     = cellfun(@(x) length(x)-1, DFenPei);

% 精英注入初始化
pop2 = elite_inject(popSz2, dim, Lb, Ub, elite_pop, elite_fit, K_elite);
fit2 = arrayfun(@(i) fobj(pop2(i,:)), 1:popSz2)';
[bestFit2, bi] = min(fit2); bestPos2=pop2(bi,:);
CC2 = zeros(maxGen2,1);

for g=1:maxGen2
    % 双目标评价
    F2 = zeros(popSz2,2);
    for i=1:popSz2
        F2(i,:) = nsga_dual_obj(pop2(i,:), DFenPei, data.dis, P_);
    end
    % 生成子代（整数版单点交叉+变异）
    child2 = zeros(popSz2,dim);
    for i=1:popSz2
        p1=pop2(randi(popSz2),:); p2=pop2(randi(popSz2),:);
        cp=randi(dim);
        child2(i,:)=[p1(1:cp-1), p2(cp:end)];
        if rand<0.08
            j=randi(dim);
            child2(i,j)=Lb(j)+randi(Ub(j)-Lb(j)+1)-1;
        end
        child2(i,:)=max(Lb,min(Ub,child2(i,:)));
    end
    % 合并父子代，按F1（总距离）排序保留前popSz2
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
fprintf('  NSGA完成 | 最优: %.2f m\n\n', bestFit2);

%% ============================================================
%  STAGE 3: NSGA — 精英池脱困扰动
%  接收Stage2精英，NSGA多样性机制帮助跳出Stage2局部最优
%% ============================================================
fprintf('[Stage 3] NSGA — 精英池脱困扰动...\n');

popSz3 = 60;
maxGen3= floor(budget.S3_NSGA / popSz3);

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
        % 增加变异率（Stage3脱困，加大扰动）
        for j=1:dim
            if rand<0.12
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
fprintf('  NSGA完成 | 最优: %.2f m\n\n', bestFit3);

%% ============================================================
%  STAGE 4: NSGA — 深度精化
%  接收Stage3精英，降低变异率，NSGA精细压缩解空间
%% ============================================================
fprintf('[Stage 4] NSGA — 深度精化...\n');

popSz4 = 60;
maxGen4= floor(budget.S4_NSGA / popSz4);

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
        % 降低变异率（Stage4精化，减小扰动）
        if rand<0.04
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
fprintf('  NSGA完成 | 最优: %.2f m\n\n', bestFit4);

%% ============================================================
%  STAGE 5: TOW — 精英注入加权绳拉深收敛
%  TOW利用种群加权中心引导，获得最多预算做最终精细收敛
%% ============================================================
fprintf('[Stage 5] TOW  — 精英注入深收敛...\n');

nT5      = 40;
maxIter5 = floor(budget.S5_TOW / nT5);
alpha5   = 0.985;
sigma05  = 2.5;

pop5 = elite_inject(nT5, dim, Lb, Ub, elite_pop, elite_fit, K_elite);
fit5 = arrayfun(@(i) fobj(pop5(i,:)), 1:nT5)';
[bestFit5, bi] = min(fit5); bestPos5=pop5(bi,:);
CC5 = zeros(maxIter5,1);

for it=1:maxIter5
    inv_f = 1./(fit5 - min(fit5) + 1e-10);
    W     = inv_f / sum(inv_f);
    wc    = W' * pop5;                    % 加权中心
    sigma = sigma05 * alpha5^it;
    for i=1:nT5
        pull = wc - pop5(i,:);
        step = round(0.65*pull + sigma*randn(1,dim));
        np   = max(Lb, min(Ub, pop5(i,:)+step));
        f    = fobj(np);
        if f < fit5(i), pop5(i,:)=np; fit5(i)=f; end
        if f < bestFit5, bestFit5=f; bestPos5=np; end
    end
    CC5(it) = bestFit5;
end

X_final = bestPos5;
CC_all=[CC_all;CC5]; stage_boundary(5)=length(CC_all);
fprintf('  TOW完成 | 最优: %.2f m\n\n', bestFit5);

%% ======================== 6. 统计输出（8项指标）========================
fprintf('========================================================\n');
fprintf('  各阶段改进汇总\n');
fprintf('========================================================\n');

stage_starts = [1, stage_boundary(1:4)+1];
for s=1:5
    seg = CC_all(stage_starts(s):stage_boundary(s));
    fprintf('  Stage%d %-5s | %12.2f → %12.2f | 改进 %.2f%%\n', ...
        s, stage_names{s}, seg(1), seg(end), ...
        100*(seg(1)-seg(end))/max(seg(1),1));
end

% 8项评价指标
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

fprintf('\n  [8项评价指标]\n');
fprintf('  TED (总疏散距离):   %.2f m\n', TED);
fprintf('  ATD (平均疏散距离): %.2f m\n', ATD);
fprintf('  MID (最大单点距离): %.2f m\n', MID);
fprintf('  SUR (避难所利用率): %.2f %%\n', SUR);
fprintf('  BTV (最终适应度):   %.4f\n',   bestFit5);
fprintf('  MET (总运行时间):   %.2f s\n', MET);
fprintf('  总迭代步数:         %d\n',     length(CC_all));
fprintf('========================================================\n');

%% ======================== 7. 可视化 ========================

% --- 图1：完整收敛曲线（5阶段分色）---
figure('Name','Pipeline Integrative 收敛','Color','w','Position',[50,50,980,460]);
hold on; box on; grid on;
for s=1:5
    sx = (stage_starts(s):stage_boundary(s))';
    sy = CC_all(stage_starts(s):stage_boundary(s));
    plot(sx, sy, 'Color', stage_colors{s}, 'LineWidth', 2.5, ...
         'DisplayName', ['Stage' num2str(s) ': ' stage_names{s}]);
    if s<5
        xline(stage_boundary(s),'--','Color',[0.6 0.6 0.6],'LineWidth',1,'HandleVisibility','off');
        text(stage_boundary(s)+5, CC_all(stage_boundary(s)), ...
             ['→' stage_names{s+1}], 'FontSize',8, 'Color',[0.35 0.35 0.35], ...
             'HandleVisibility','off');
    end
end
xlabel('Cumulative Iteration', 'FontWeight','bold');
ylabel('Best Fitness — Total Distance (m)', 'FontWeight','bold');
title('Pipeline Integrative Hybrid (ABC→NSGA×3→TOW)', 'FontSize',13,'FontWeight','bold');
legend('Location','northeast');

% --- 图2：各阶段贡献柱状图 ---
figure('Name','各阶段贡献','Color','w','Position',[200,200,680,430]);
impr = zeros(1,5);
for s=1:5
    seg = CC_all(stage_starts(s):stage_boundary(s));
    impr(s) = max(seg(1)-seg(end), 0);
end
b = bar(impr, 'FaceColor','flat');
for s=1:5, b.CData(s,:) = stage_colors{s}; end
set(gca, 'XTickLabel', ...
    {'S1:ABC','S2:NSGA','S3:NSGA','S4:NSGA','S5:TOW'}, 'FontSize',11);
xlabel('Algorithm Stage', 'FontWeight','bold');
ylabel('Fitness Improvement (m)', 'FontWeight','bold');
title('Contribution per Stage — Pipeline Integrative','FontSize',12,'FontWeight','bold');
grid on; box on;
for s=1:5
    text(s, impr(s)+max(impr)*0.015, sprintf('%.0f', impr(s)), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
end

% --- 图3：2D 疏散分配地图 ---
figure('Color','w','Name','Pipeline Integrative Final Map','Position',[100,100,900,800]);
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
title('Pipeline Integrative Optimized Allocation Map','FontSize',13,'FontWeight','bold');
legend([h_shl,h_res],{'Shelter','Residential'},'Location','northeast');

fprintf('\n[完成] Pipeline Integrative 运行结束。\n');

%% ============================================================
%  子函数
%% ============================================================

% 统一目标函数（整数编码，总疏散距离）
function fitness = unified_fobj(x, DFenPei, dis_mat, Lb, Ub)
    fitness=0;
    for i=1:length(DFenPei)
        idx=max(Lb(i), min(round(x(i)), Ub(i)));
        fitness=fitness+dis_mat(DFenPei{i}(1), DFenPei{i}(idx+1));
    end
end

% NSGA双目标：[总疏散距离, 避难所容量方差]
function f2 = nsga_dual_obj(x, DFenPei, dis_mat, P_)
    X   = max(1, min(round(x), P_));
    td  = 0;
    Y   = zeros(1, size(dis_mat,2));
    for i=1:length(X)
        hid=DFenPei{i}(1); eid=DFenPei{i}(X(i)+1);
        td=td+dis_mat(hid,eid);
        Y(eid)=Y(eid)+12;
    end
    f2=[td, var(Y)];
end

% 精英注入初始化：前K_elite个直接继承，其余在精英附近扰动
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