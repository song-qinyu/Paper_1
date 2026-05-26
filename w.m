%% ============================================================
%  Pipeline Hybrid — Stage 竞选框架 v2（全15算法版）
%
%  参与竞选的12种算法（统一整数编码，同一目标函数）：
%    GA, DE, PSO, SSA, GWO, FA, ABC, TOW, CA, PO, CS, HLO
%
%  接口适配说明（以下3种问题结构不同，独立展示）：
%    SA   —— 选址问题（选57个中心），与分配优化不同
%    HS   —— K-means聚类式选址，非逐点分配
%    NSGA —— 多目标，返回Pareto前沿，无单一最优值
%    以上三种不纳入Stage竞选，但可作为对比参考算法单独展示
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