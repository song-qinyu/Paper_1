% 引力搜索优化算法 
function [Best_pos,Best_fitness,Iter_curve, History_pos, History_best]=GSA(pop, maxIter, lb, ub, dim,fobj)
%input
%pop 种群数量
%dim 问题维数
%ub 变量上边界
%lb 变量下边界
%fobj 适应度函数
%maxIter 最大迭代次数
%output
%Best_pos 最优位置
%Best_fitness 最优适应度值
%Iter_curve 每代最优适应度值
%History_pos 每代种群位置
%History_best 每代最优个体位置
%% 初始化种群位置
X = initialization(pop, ub, lb, dim);
%% 计算适应度值
for i = 1:pop
    fitness(i) = fobj(X(i,:));
end
% 最优位置&最优适应度值
[SortFitness, indexSort] = sort(fitness);
gBest = X(indexSort(1),:);
gBestFitness = SortFitness(1);
M = zeros(pop, 1); %质量矩阵
V = zeros(pop,dim); %速度矩阵
%% 迭代
for t = 1:maxIter
    %计算质量
    M = massCalculation(fitness);
    %计算引力常数
    G = Gconstant(t, maxIter);
    %计算加速度
    a = Acceleration(M,X,G,t,maxIter);
    %位置更新
    [X,V] = move(X,a,V);
    %边界检查
    Flag4ub=X(i,:)>ub;
    Flag4lb=X(i,:)<lb;
    X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
    
    %计算适应度值
    for i = 1:pop
        fitness(i) = fobj(X(i,:));
        if fitness(i) < gBestFitness
            gBestFitness = fitness(i);
            gBest = X(i,:);
        end
    end
    History_pos{t} = X;
    History_best{t} = gBest;
    Iter_curve(t) = gBestFitness;
end
Best_pos = gBest;
Best_fitness = gBestFitness;
end
%% 初始化函数
function X=initialization(SearchAgents_no,ub,lb,dim)

Boundary_no= size(ub,2); 
if Boundary_no==1
    X=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        X(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end
end
%% 计算质量
function M = massCalculation(fitness)
    bestF = min(fitness);
    worstF = max(fitness);
    M = (fitness - worstF) ./ (bestF - worstF);
    M = M ./ sum(M);
end
%% 引力常数计算
function G = Gconstant(iter, max_it)
    alfa = 20;
    G0 = 100;
    G = G0*exp(-alfa*iter/max_it);
end
%% 计算加速度
function a = Acceleration(M, X, G, iter, max_it)
    [N, dim] = size(X);
    final_per = 2;
    kbest = final_per + (1 - iter/max_it) * (100-final_per);
    kbest = round(N*kbest/100);
    [Ms, ds] = sort(M, 'descend');
    for i = 1:N
        F(i,:) = zeros(1,dim);
        for ii = 1:kbest
            j = ds(ii);
            if j ~= i
                R = norm(X(i,:) - X(j,:), 2);
                for k = 1:dim
                    F(i,k) = F(i,k) + rand*M(j)*((X(j,k) - X(i,k))/(R+eps));
                end
            end
        end
    end
    a = F.*G;
end
%% 位置更新
function [X,V] = move(X,a,V)
    [N,dim] = size(X);
    V = rand(N,dim).*V+a;
    X = X + V;
end