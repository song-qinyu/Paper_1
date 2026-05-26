%% ============================================================
%  Pipeline Hybrid — Stage 竞选框架 v2（全15算法版）
%
%  全部15种算法均参与竞选，统一用 unified_fobj（整数编码，总疏散距离）评价
%
%  直接参与的12种算法：GA, DE, PSO, SSA, GWO, FA, ABC, TOW, CA, PO, CS, HLO
%
%  后处理适配的3种算法（原始结构不同，结果转换后纳入统一排名）：
%    SA   —— 原为选址 → 将SA选出中心映射为DFenPei整数编码 → unified_fobj
%    HS   —— 原为聚类坐标优化 → 匹配最近真实避难所 → 整数编码 → unified_fobj
%    NSGA —— 原为多目标 → 取Pareto前沿中F1最小解 → unified_fobj
%
%  公平性保证：
%    1. 所有算法：同一目标函数 unified_fobj（整数编码）
%    2. 所有算法：每阶段相同 NFE（函数评价次数）预算
%    3. 所有算法：接收同一精英池作为初始种群
%    4. 胜者由数据决定，不由研究者主观指定
%% ============================================================
clc; clear; close all; tic;

fprintf('========================================================\n');
fprintf('  Pipeline Stage 竞选框架 v2 — 全12算法公平评测\n');
fprintf('========================================================\n\n');

%% ==================== 0. 数据加载 ====================
if exist('sj5.mat','file'), load('sj5.mat');
else, error('未找到 sj5.mat'); end

if exist('dis','var'), data.dis = dis; end

dim = length(DFenPei);
Lb  = ones(1, dim);
Ub  = arrayfun(@(i) length(DFenPei{i})-1, 1:dim);

% 固定分配点
FID=[]; alldis_fixed=0;
for k=1:length(B)
    if length(B{k})==1
        FID=[FID;k,B{k}];
        alldis_fixed=alldis_fixed+data.dis(k,B{k});
    end
end

% 统一目标函数（整数编码）
fobj = @(x) unified_fobj(x, DFenPei, data.dis, Lb, Ub);

%% ==================== 1. 预算设置 ====================
% 每阶段给每个算法相同的函数评价次数
% 建议调试时用 5000，正式运行用 20000+
NFE_per_algo = 15000;
K_elite = 20;  % 精英池大小

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

%% ==================== 3. 汇总报告 ====================
fprintf('========================================================\n');
fprintf('  各阶段最优算法汇总\n');
fprintf('========================================================\n');
for s=1:5
    r = all_results{s};
    fits = [r.best_fit];
    [~,wi] = min(fits);
    fprintf('  Stage %d (%s): 最优算法 = %-6s | 最优值 = %.2f m\n', ...
        s, stage_labels{s}, r(wi).name, r(wi).best_fit);
end

fprintf('\n建议的 Pipeline 组合（数据驱动）：\n');
for s=1:5
    r = all_results{s};
    [~,wi] = min([r.best_fit]);
    fprintf('  Stage %d → %s\n', s, r(wi).name);
end
fprintf('\n总评测时间：%.2f 秒\n', toc);
fprintf('========================================================\n');

%% ==================== 4. 可视化 ====================
figure('Color','w','Name','各阶段竞选结果','Position',[50,50,1400,820]);
for s=1:5
    subplot(2,3,s);
    r = all_results{s};
    names = {r.name};
    fits  = [r.best_fit];
    [fits_s, si] = sort(fits);
    names_s = names(si);

    b = barh(fits_s, 'FaceColor','flat');
    clrs = repmat([0.75 0.88 1.0], length(fits_s),1);
    clrs(1,:) = [0.18 0.72 0.32];  % 第一名绿色
    clrs(2,:) = [0.95 0.85 0.20];  % 第二名黄色
    b.CData = clrs;

    yticks(1:length(names_s));
    yticklabels(names_s);
    xlabel('适应度（总距离 m）');
    title(sprintf('Stage %d  %s', s, stage_labels{s}), 'FontSize',11,'FontWeight','bold');
    grid on; box on;

    text(fits_s(1)*1.001, 1, sprintf(' %.0f ★', fits_s(1)), ...
        'FontSize',8,'Color',[0.05 0.45 0.1],'FontWeight','bold');
end

% 子图6：各阶段最优值折线（pipeline收敛趋势）
subplot(2,3,6);
stage_bests = zeros(1,5);
stage_winners = cell(1,5);
for s=1:5
    r=all_results{s}; [v,wi]=min([r.best_fit]);
    stage_bests(s)=v; stage_winners{s}=r(wi).name;
end
plot(1:5, stage_bests, '-o','LineWidth',2.5,'Color',[0.2 0.5 0.9],...
    'MarkerFaceColor',[0.2 0.5 0.9],'MarkerSize',8);
xticks(1:5);
xticklabels(cellfun(@(s,w) sprintf('S%d\n%s',s,w), ...
    num2cell(1:5), stage_winners, 'UniformOutput',false));
ylabel('最优适应度 (m)'); title('Pipeline 各阶段最优值趋势','FontSize',11,'FontWeight','bold');
grid on; box on;

sgtitle('Pipeline Hybrid 各阶段算法竞选结果（越短越优）','FontSize',13,'FontWeight','bold');

%% ============================================================
%  核心调度函数
%% ============================================================
function results = run_stage_competition(fobj, dim, Lb, Ub, NFE, elite_pop, elite_fit, K_elite)
    idx = 0;
    results = struct('name',{},'best_fit',{},'best_pos',{},'pop',{},'fit',{});

    algo_list = {
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
        'SA',   @run_SA;     % 选址→最近邻分配→转fobj
        'HS',   @run_HS;     % 聚类→最近邻分配→转fobj
        'NSGA', @run_NSGA;   % 多目标→取总距离最小解→转fobj
    };

    for a = 1:size(algo_list,1)
        name = algo_list{a,1};
        fn   = algo_list{a,2};
        fprintf('  评测 %-4s ... ', name);
        try
            [bf, bp, pop_out, fit_out] = fn(fobj, dim, Lb, Ub, NFE, elite_pop, elite_fit, K_elite);
            idx=idx+1;
            results(idx).name     = name;
            results(idx).best_fit = bf;
            results(idx).best_pos = bp;
            results(idx).pop      = pop_out;
            results(idx).fit      = fit_out;
            fprintf('最优: %.2f\n', bf);
        catch ME
            fprintf('[错误跳过] %s\n', ME.message);
        end
    end
end

function print_ranking(results)
    [fits_s, si] = sort([results.best_fit]);
    fprintf('  排名 | 算法  | 最优适应度\n');
    fprintf('  -----|-------|--------------------\n');
    for r=1:length(si)
        mark=''; if r==1, mark=' ← 胜出'; end
        fprintf('  %3d  | %-4s  | %.2f m%s\n', r, results(si(r)).name, fits_s(r), mark);
    end
end

function [ep, ef] = get_elite_pool(results, K, dim)
    all_pop=[]; all_fit=[];
    for i=1:length(results)
        if ~isempty(results(i).pop) && ~isempty(results(i).fit)
            all_pop=[all_pop; results(i).pop];
            all_fit=[all_fit; results(i).fit(:)];
        end
    end
    if isempty(all_pop), ep=[]; ef=[]; return; end
    [~,si]=sort(all_fit);
    K=min(K,length(si));
    ep=all_pop(si(1:K),:);
    ef=all_fit(si(1:K));
end

function pop = init_pop(popSize, dim, Lb, Ub, elite_pop, K_elite)
    pop = zeros(popSize, dim);
    ne  = min(size(elite_pop,1), K_elite);
    for i=1:popSize
        if i<=ne && ne>0
            pop(i,:) = elite_pop(i,:);
        elseif ne>0
            base  = elite_pop(randi(ne),:);
            noise = round(randn(1,dim) .* max(1,(Ub-Lb)*0.04));
            pop(i,:) = max(Lb, min(Ub, base+noise));
        else
            pop(i,:) = Lb + round(rand(1,dim).*(Ub-Lb));
        end
    end
end

%% ============================================================
%  12个算法封装（统一接口）
%  [best_fit, best_pos, pop, fit] = run_XXX(fobj, dim, Lb, Ub, NFE, elite_pop, elite_fit, K_elite)
%% ============================================================

%% ---- 1. GA（遗传算法）----
function [best_fit, best_pos, pop, fit] = run_GA(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    popSize = 60; maxGen = floor(NFE/popSize);
    pc=0.85; pm=0.12;
    pop = init_pop(popSize, dim, Lb, Ub, ep, K);
    fit = arrayfun(@(i) fobj(pop(i,:)), 1:popSize)';
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);

    for iter_=1:maxGen
        % 锦标赛选择
        newP=zeros(size(pop));
        for i=1:popSize
            c=randperm(popSize,3); [~,w]=min(fit(c));
            newP(i,:)=pop(c(w),:);
        end
        % 单点交叉
        for i=1:2:popSize-1
            if rand<pc
                cp=randi(dim);
                tmp=newP(i,cp:end); newP(i,cp:end)=newP(i+1,cp:end); newP(i+1,cp:end)=tmp;
            end
        end
        % 整数变异（随机跳到合法值）
        for i=1:popSize
            for j=1:dim
                if rand<pm
                    newP(i,j)=Lb(j)+randi(Ub(j)-Lb(j)+1)-1;
                end
            end
        end
        newP=max(Lb,min(Ub,newP));
        pop=newP;
        for i=1:popSize
            fit(i)=fobj(pop(i,:));
            if fit(i)<best_fit, best_fit=fit(i); best_pos=pop(i,:); end
        end
        pop(1,:)=best_pos; fit(1)=best_fit;  % 精英保留
    end
end

%% ---- 2. DE（差分进化）----
function [best_fit, best_pos, pop, fit] = run_DE(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    popSize=50; maxGen=floor(NFE/popSize);
    F=0.7; CR=0.8;
    pop=init_pop(popSize, dim, Lb, Ub, ep, K);
    fit=arrayfun(@(i) fobj(pop(i,:)), 1:popSize)';
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

%% ---- 3. PSO（粒子群）----
function [best_fit, best_pos, pop, fit] = run_PSO(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    popSize=50; maxIter=floor(NFE/popSize);
    w=0.8; c1=1.5; c2=1.5;
    pop=init_pop(popSize, dim, Lb, Ub, ep, K);
    vel=zeros(popSize,dim);
    fit=arrayfun(@(i) fobj(pop(i,:)), 1:popSize)';
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

%% ---- 4. SSA（麻雀搜索）----
function [best_fit, best_pos, pop, fit] = run_SSA(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    popSize=50; maxIter=floor(NFE/popSize); P_pct=0.2;
    pop=double(init_pop(popSize, dim, Lb, Ub, ep, K));
    fit=arrayfun(@(i) fobj(pop(i,:)), 1:popSize)';
    pFit=fit; pX=pop;
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);

    for t=1:maxIter
        [~,sortIdx]=sort(fit); [~,wIdx]=max(fit);
        worse=pop(wIdx,:); nd=round(popSize*P_pct);
        for i=1:nd
            si=sortIdx(i);
            if rand<0.8, pop(si,:)=round(pX(si,:).*exp(-i/(rand*maxIter)));
            else,        pop(si,:)=round(pX(si,:)+randn(1,dim)); end
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
            pop(i,:)=max(Lb,min(Ub,pop(i,:)));
            fit(i)=fobj(pop(i,:));
            if fit(i)<pFit(i), pFit(i)=fit(i); pX(i,:)=pop(i,:); end
            if pFit(i)<best_fit, best_fit=pFit(i); best_pos=pX(i,:); end
        end
    end
end

%% ---- 5. GWO（灰狼优化）----
function [best_fit, best_pos, pop, fit] = run_GWO(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    n=50; maxIter=floor(NFE/n);
    pop=init_pop(n, dim, Lb, Ub, ep, K);
    fit=arrayfun(@(i) fobj(pop(i,:)), 1:n)';
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

%% ---- 6. FA（萤火虫）----
function [best_fit, best_pos, pop, fit] = run_FA(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    n=40; maxIter=floor(NFE/n);
    alpha=0.5; betamin=0.2; gamma=1;
    pop=double(init_pop(n, dim, Lb, Ub, ep, K));
    fit=arrayfun(@(i) fobj(pop(i,:)), 1:n)';
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

%% ---- 7. ABC（人工蜂群）----
function [best_fit, best_pos, pop, fit] = run_ABC(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    FN=25; limit=20; maxCycle=floor(NFE/(FN*2));
    pop=init_pop(FN, dim, Lb, Ub, ep, K);
    fit=arrayfun(@(i) fobj(pop(i,:)), 1:FN)';
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

%% ---- 8. TOW（拔河优化）----
function [best_fit, best_pos, pop, fit] = run_TOW(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    nT=30; maxIter=floor(NFE/nT);
    alpha_t=0.98; sigma0=2.0;
    pop=double(init_pop(nT, dim, Lb, Ub, ep, K));
    fit=arrayfun(@(i) fobj(pop(i,:)), 1:nT)';
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);

    for it=1:maxIter
        inv_f=1./(fit-min(fit)+1e-10);
        W=inv_f/sum(inv_f);
        wc=W'*pop;
        sigma=sigma0*alpha_t^it;
        for i=1:nT
            step=round(0.6*(wc-pop(i,:))+sigma*randn(1,dim));
            np=max(Lb,min(Ub,pop(i,:)+step));
            f=fobj(np);
            if f<fit(i), pop(i,:)=np; fit(i)=f; end
            if f<best_fit, best_fit=f; best_pos=np; end
        end
    end
end

%% ---- 9. CA（文化算法）——整合 u2.m 逻辑 ----
function [best_fit, best_pos, pop, fit] = run_CA(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    nPop=50; nAccept=round(0.2*nPop); alpha_ca=0.15;
    maxIter=floor(NFE/nPop);
    pop=double(init_pop(nPop, dim, Lb, Ub, ep, K));
    fit=arrayfun(@(i) fobj(pop(i,:)), 1:nPop)';
    [~,si]=sort(fit);
    best_fit=fit(si(1)); best_pos=pop(si(1),:);

    % 信仰空间（Normative + Situational）
    cult_best=best_pos;
    norm_lo=Lb; norm_hi=Ub;

    for iter_=1:maxIter
        for i=1:nPop
            sigma=alpha_ca*(norm_hi-norm_lo);
            dx=round(sigma.*randn(1,dim));
            % 向 Situational 方向偏移
            dx=dx+round(0.3*sign(cult_best-pop(i,:)));
            pop(i,:)=max(Lb,min(Ub,round(pop(i,:)+dx)));
            fit(i)=fobj(pop(i,:));
        end
        [~,si]=sort(fit);
        % 更新 Normative 空间（取前 nAccept 个）
        spop=pop(si(1:nAccept),:);
        norm_lo=min(spop,[],1);
        norm_hi=max(spop,[],1)+1;  % 防止退化为零宽度
        if fit(si(1))<best_fit
            best_fit=fit(si(1)); best_pos=pop(si(1),:);
            cult_best=best_pos;
        end
    end
end

%% ---- 10. PO（政治优化）——内联 Main_PO_Run.m 逻辑 ----
function [best_fit, best_pos, pop, fit] = run_PO(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    parties=10; areas=10;
    popSize=parties*areas;  % 100
    maxIter=floor(NFE/popSize);

    % 初始化种群（精英注入）
    pop=init_pop(popSize, dim, Lb, Ub, ep, K);
    fit=arrayfun(@(i) fobj(pop(i,:)), 1:popSize)';
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);

    for it=1:maxIter
        % ---- 党内学习（每党 areas 个成员向本党领袖靠拢）----
        for p=1:parties
            idx_p=(p-1)*areas+1:p*areas;
            [~,ll]=min(fit(idx_p));
            leader_idx=idx_p(ll);
            leader=pop(leader_idx,:);
            for m=idx_p
                if m==leader_idx, continue; end
                lr=0.3+0.4*rand;
                step=round((leader-pop(m,:))*lr + randn(1,dim)*0.8);
                np=max(Lb,min(Ub,pop(m,:)+step));
                f=fobj(np);
                if f<fit(m), pop(m,:)=np; fit(m)=f; end
                if f<best_fit, best_fit=f; best_pos=np; end
            end
        end
        % ---- 跨党竞争（各党领袖向全局最优靠拢）----
        [~,gsi]=sort(fit);
        global_leader=pop(gsi(1),:);
        for p=1:parties
            idx_p=(p-1)*areas+1:p*areas;
            [~,ll]=min(fit(idx_p)); li=idx_p(ll);
            step=round((global_leader-pop(li,:))*0.25*rand+randn(1,dim)*0.6);
            np=max(Lb,min(Ub,pop(li,:)+step));
            f=fobj(np);
            if f<fit(li), pop(li,:)=np; fit(li)=f; end
            if f<best_fit, best_fit=f; best_pos=np; end
        end
    end
end

%% ---- 11. CS（布谷鸟搜索）----
function [best_fit, best_pos, pop, fit] = run_CS(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    n=30; maxIter=floor(NFE/n);
    pa=0.25; beta=1.5;

    pop=init_pop(n, dim, Lb, Ub, ep, K);
    fit=arrayfun(@(i) fobj(pop(i,:)), 1:n)';
    [best_fit,bi]=min(fit); best_pos=pop(bi,:);

    for iter_=1:maxIter
        % Lévy 飞行更新
        for i=1:n
            levy=levy_cs(beta, dim);
            scale=max(1, round(0.05*mean(Ub-Lb)));
            np=max(Lb,min(Ub, round(pop(i,:)+scale*levy.*(pop(i,:)-best_pos)+randn(1,dim))));
            f=fobj(np);
            j=randi(n);
            if f<fit(j), pop(j,:)=np; fit(j)=f; end
            if f<best_fit, best_fit=f; best_pos=np; end
        end
        % 废巢（随机丢弃 pa 比例的巢）
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

function levy = levy_cs(beta, dim)
    sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u=randn(1,dim)*sigma; v=randn(1,dim);
    levy=u./(abs(v).^(1/beta));
    levy=sign(levy).*min(abs(levy),5);
end

%% ---- 12. HLO（人类学习优化，二进制编码适配版）----
function [best_fit, best_pos, pop, fit] = run_HLO(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    popSize=40; bpv=8;  % 每维 8 位二进制，精度 = (Ub-Lb)/255
    m=dim*bpv;
    maxIter=floor(NFE/popSize);
    p_r=0.1; p_i=0.5;

    % 初始化二进制种群
    % 若有精英池，将精英编码为二进制作为初始解
    bin_pop=zeros(popSize, m);
    for i=1:popSize
        if i<=min(size(ep,1),K) && size(ep,1)>0
            % 将整数解编码为二进制
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

    % 解码函数：二进制 → 整数解
    decode = @(row) arrayfun(@(v) ...
        max(Lb(v), min(Ub(v), Lb(v)+round( ...
            sum(row((v-1)*bpv+1:v*bpv).*(2.^(bpv-1:-1:0)))/(2^bpv-1) ...
            *(Ub(v)-Lb(v))))), 1:dim);

    IKD=bin_pop;
    IKDfits=arrayfun(@(i) fobj(decode(bin_pop(i,:))), 1:popSize)';
    [best_val,bi]=min(IKDfits);
    SKD=IKD(bi,:);
    SKDfit=best_val;
    best_pos=decode(SKD);
    best_fit=SKDfit;

    % 存整数种群用于输出
    pop_int=zeros(popSize,dim);
    for i=1:popSize, pop_int(i,:)=decode(bin_pop(i,:)); end

    for iter_=1:maxIter
        for i=1:popSize
            for j=1:m
                pr=rand;
                if pr<p_r,       bin_pop(i,j)=randi([0,1]);
                elseif pr<p_i,   bin_pop(i,j)=IKD(i,j);
                else,            bin_pop(i,j)=SKD(j); end
            end
            x_int=decode(bin_pop(i,:));
            fv=fobj(x_int);
            if fv<IKDfits(i), IKDfits(i)=fv; IKD(i,:)=bin_pop(i,:); end
            if fv<SKDfit, SKDfit=fv; SKD=bin_pop(i,:); best_pos=x_int; best_fit=fv; end
        end
        for i=1:popSize, pop_int(i,:)=decode(bin_pop(i,:)); end
    end

    pop=pop_int;
    fit=IKDfits;
end

%% ---- 13. SA（模拟退火）——选址→最近邻→后处理转fobj ----
% SA 原始逻辑：从所有候选避难所中选出若干个中心
% 后处理：将每个住宅点（DFenPei候选点）就近分配到SA选出的中心
% 即：对每个 DFenPei{i} 的候选列表，找与SA中心最近的那个索引 → 转为合法整数解
function [best_fit, best_pos, pop, fit] = run_SA(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    % 获取数据：需从工作空间读取 DFenPei 和 dis_mat
    % 因为SA需要知道避难所坐标/距离来选中心
    % 这里用 fobj 的闭包变量无法直接访问，因此通过 evalin 从 base 读取
    try
        DFenPei_  = evalin('base','DFenPei');
        dis_mat_  = evalin('base','data.dis');
    catch
        % 若无法读取则退回随机解
        pop = init_pop(20, dim, Lb, Ub, ep, K);
        fit = arrayfun(@(i) fobj(pop(i,:)), 1:20)';
        [best_fit,bi] = min(fit); best_pos = pop(bi,:);
        return;
    end

    % SA参数
    T0=1e5; Tend=1; q=0.93; L=30;
    NFE_used=0;

    % 收集所有候选避难所索引（DFenPei{i}(2:end) 是候选避难所）
    all_cands = unique(cell2mat(cellfun(@(c) c(2:end), DFenPei_,'UniformOutput',false)));
    nCands = length(all_cands);
    num_sel = min(57, nCands);  % 选num_sel个中心（与原SA一致）

    % 初始解：随机选 num_sel 个候选避难所
    S1 = all_cands(randperm(nCands, num_sel));
    x_int1 = sa_to_fobj_x(S1, DFenPei_, Lb, Ub);
    best_fit = fobj(x_int1); NFE_used=NFE_used+1;
    best_S = S1; best_pos = x_int1;

    while T0>Tend && NFE_used<NFE
        for i=1:L
            % 生成邻域解：随机替换1个选中的中心
            S2=S1; unsel=setdiff(all_cands,S1);
            if isempty(unsel), break; end
            S2(randi(num_sel))=unsel(randi(length(unsel)));
            x_int2=sa_to_fobj_x(S2, DFenPei_, Lb, Ub);
            f2=fobj(x_int2); NFE_used=NFE_used+1;
            delta=f2-best_fit;
            if f2<best_fit || exp(-delta/T0)>rand
                S1=S2;
                if f2<best_fit
                    best_fit=f2; best_S=S2; best_pos=x_int2;
                end
            end
            if NFE_used>=NFE, break; end
        end
        T0=T0*q;
    end

    % 输出整数种群（SA只有单解，复制成小种群供精英池使用）
    pop_size=min(20,dim);
    pop=repmat(best_pos,pop_size,1);
    % 对每行加小扰动
    for i=2:pop_size
        pop(i,:)=max(Lb,min(Ub,best_pos+round(randn(1,dim))));
    end
    fit=arrayfun(@(i) fobj(pop(i,:)), 1:pop_size)';
end

function x_int = sa_to_fobj_x(sel_centers, DFenPei_, Lb, Ub)
% 将SA选出的避难所集合，转换为统一整数编码
% 对每个 DFenPei_{i}，在候选列表中找与sel_centers重合的最近那个的索引
    dim_=length(DFenPei_);
    x_int=zeros(1,dim_);
    for i=1:dim_
        cands=DFenPei_{i}(2:end);   % 候选避难所列表
        % 找 cands 中属于 sel_centers 的那个，若有多个取第一个
        overlap=intersect(cands, sel_centers);
        if ~isempty(overlap)
            idx=find(DFenPei_{i}==overlap(1))-1;  % 在DFenPei中的偏移量
        else
            % 若无重合，取第一个候选
            idx=1;
        end
        x_int(i)=max(Lb(i), min(Ub(i), idx));
    end
end

%% ---- 14. HS（和声搜索）——聚类选址→最近邻→后处理转fobj ----
% HS 原始逻辑：优化避难所坐标位置（连续坐标）
% 后处理：把HS求出的坐标与真实避难所坐标匹配，得到最近的合法避难所索引
function [best_fit, best_pos, pop, fit] = run_HS(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    try
        DFenPei_ = evalin('base','DFenPei');
        data_     = evalin('base','data');
        dis_mat_  = data_.dis;
        binan_xy  = data_.binan;  % 真实避难所坐标矩阵 [n×2]
        start_xy  = data_.start;  % 住宅坐标
    catch
        pop = init_pop(20, dim, Lb, Ub, ep, K);
        fit = arrayfun(@(i) fobj(pop(i,:)), 1:20)';
        [best_fit,bi] = min(fit); best_pos = pop(bi,:);
        return;
    end

    % HS参数
    HMS=15; num_centers=min(57,size(binan_xy,1));
    house_x=start_xy(:,1); house_y=start_xy(:,2);
    min_x=min(house_x); max_x=max(house_x);
    min_y=min(house_y); max_y=max(house_y);
    NVAR=num_centers*2;
    NFE_used=0;

    BW_max=(max_x-min_x)*0.2; BW_min=(max_x-min_x)*0.0001;
    maxItr=floor(NFE/HMS);

    % 初始化和声记忆库
    HM=zeros(HMS,NVAR); hm_fit=zeros(HMS,1);
    for i=1:HMS
        idx_r=randperm(size(house_x,1),num_centers);
        pos=[house_x(idx_r), house_y(idx_r)];
        HM(i,:)=pos(:)';
        hm_fit(i)=hs_eval(HM(i,:), num_centers, house_x, house_y, fobj, DFenPei_, binan_xy, Lb, Ub);
        NFE_used=NFE_used+1;
    end
    [best_hf,bi]=min(hm_fit);
    best_harm=HM(bi,:);

    for itr=1:maxItr
        if NFE_used>=NFE, break; end
        BW=BW_max*exp(log(BW_min/BW_max)*itr/maxItr);
        [~,bi_]=min(hm_fit);
        new_h=HM(bi_,:);
        t=randi(num_centers); ix=t*2-1; iy=t*2;
        new_h(ix)=new_h(ix)+(rand*2-1)*BW;
        new_h(iy)=new_h(iy)+(rand*2-1)*BW;
        new_h(ix)=max(min(new_h(ix),max_x),min_x);
        new_h(iy)=max(min(new_h(iy),max_y),min_y);
        fnew=hs_eval(new_h, num_centers, house_x, house_y, fobj, DFenPei_, binan_xy, Lb, Ub);
        NFE_used=NFE_used+1;
        [worst_f,wi]=max(hm_fit);
        if fnew<worst_f
            HM(wi,:)=new_h; hm_fit(wi)=fnew;
            if fnew<best_hf, best_hf=fnew; best_harm=new_h; end
        end
    end

    % 将最优和声解转为整数编码
    best_pos=hs_harm_to_int(best_harm, num_centers, DFenPei_, binan_xy, Lb, Ub);
    best_fit=fobj(best_pos);

    pop_size=min(20,HMS);
    pop=repmat(best_pos,pop_size,1);
    for i=2:pop_size
        pop(i,:)=max(Lb,min(Ub,best_pos+round(randn(1,dim))));
    end
    fit=arrayfun(@(i) fobj(pop(i,:)), 1:pop_size)';
end

function fv = hs_eval(harm, nc, hx, hy, fobj, DFenPei_, binan_xy, Lb, Ub)
    x_int=hs_harm_to_int(harm, nc, DFenPei_, binan_xy, Lb, Ub);
    fv=fobj(x_int);
end

function x_int = hs_harm_to_int(harm, nc, DFenPei_, binan_xy, Lb, Ub)
% 将HS的连续坐标方案映射为整数编码
% 思路：HS给出nc个中心坐标→找最近真实避难所→再对每个DFenPei点选最近那个
    centers=reshape(harm,[],2);  % nc×2
    n_binan=size(binan_xy,1);
    % 找每个中心最近的真实避难所
    sel=zeros(1,nc);
    for c=1:nc
        d=sqrt((binan_xy(:,1)-centers(c,1)).^2+(binan_xy(:,2)-centers(c,2)).^2);
        [~,mi]=min(d); sel(c)=mi;
    end
    sel=unique(sel);
    % 再转为fobj整数编码（与SA后处理相同逻辑）
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

%% ---- 15. NSGA-II（非支配排序遗传）——多目标→取总距离最小解→fobj ----
% NSGA原始：双目标（总距离 + 容量方差）返回Pareto前沿
% 适配：从Pareto前沿中选总距离最小的解作为代表，纳入统一排名
function [best_fit, best_pos, pop, fit] = run_NSGA(fobj, dim, Lb, Ub, NFE, ep, ef, K)
    try
        DFenPei_ = evalin('base','DFenPei');
        data_     = evalin('base','data');
        dis_mat_  = data_.dis;
    catch
        pop = init_pop(20, dim, Lb, Ub, ep, K);
        fit = arrayfun(@(i) fobj(pop(i,:)), 1:20)';
        [best_fit,bi] = min(fit); best_pos = pop(bi,:);
        return;
    end

    popSize=min(100, floor(NFE/20));  % 控制种群以不超预算
    maxGen=floor(NFE/popSize);
    P_  = cellfun(@(x) length(x)-1, DFenPei_);

    % 双目标函数
    nsga_obj = @(x) nsga_eval(x, DFenPei_, dis_mat_, P_);

    % 初始化种群（整数编码，与统一接口一致）
    if ~isempty(ep)
        pop=init_pop(popSize, dim, Lb, Ub, ep, K);
    else
        pop=Lb+round(rand(popSize,dim).*(Ub-Lb));
    end

    % 计算初始双目标值
    F=zeros(popSize,2);
    for i=1:popSize, F(i,:)=nsga_obj(pop(i,:)); end

    for g=1:maxGen
        % 生成子代（SBX交叉 + 多项式变异，这里简化为整数版）
        child=zeros(popSize,dim);
        for i=1:popSize
            p1=pop(randi(popSize),:);
            p2=pop(randi(popSize),:);
            cp=randi(dim);
            child(i,:)=[p1(1:cp-1), p2(cp:end)];
            if rand<0.1
                child(i,randi(dim))=Lb(randi(dim))+randi(max(1,Ub(randi(dim))-Lb(randi(dim))+1))-1;
            end
            child(i,:)=max(Lb,min(Ub,child(i,:)));
        end
        % 合并父代+子代
        combined=[pop;child];
        Fc=zeros(2*popSize,2);
        for i=1:2*popSize, Fc(i,:)=nsga_obj(combined(i,:)); end
        % 非支配排序（快速版：只保留前popSize个）
        [~,si]=sort(Fc(:,1));  % 先按F1排序（总距离）
        pop=combined(si(1:popSize),:);
        F=Fc(si(1:popSize),:);
    end

    % 从最终种群中取总距离最小的解
    [~,bi]=min(F(:,1));
    best_pos=pop(bi,:);
    best_fit=fobj(best_pos);   % 用统一fobj评价（单目标），保证可比性

    fit=arrayfun(@(i) fobj(pop(i,:)), 1:popSize)';
end

function f2 = nsga_eval(x, DFenPei_, dis_mat_, P_)
    X=max(1,min(round(x),P_));
    total_d=0; Y=zeros(1,size(dis_mat_,2));
    for i=1:length(X)
        hid=DFenPei_{i}(1); eid=DFenPei_{i}(X(i)+1);
        total_d=total_d+dis_mat_(hid,eid);
        Y(eid)=Y(eid)+12;
    end
    f2=[total_d, var(Y)];
end

%% ============================================================
%  统一目标函数
%% ============================================================
function fitness = unified_fobj(x, DFenPei, dis_mat, Lb, Ub)
    fitness=0;
    for i=1:length(DFenPei)
        idx=max(Lb(i), min(round(x(i)), Ub(i)));
        fitness=fitness+dis_mat(DFenPei{i}(1), DFenPei{i}(idx+1));
    end
end
