function EvacuationNetworkDemo()
    clc; clear; close all;

    %% ============ 1. 配置参数 ============
    [buildingData, exitData, netEdges, vMax, alpha, beta, forceAssign] = SetupParameters();

    nBuilding = length(buildingData);
    nExit     = length(exitData);

    % 将"楼栋" 和 "出口" 合并视为网络节点
    % 楼栋ID=1..nBuilding, 出口ID=nBuilding+1..nBuilding+nExit
    nNode = nBuilding + nExit;

    % 构建有向图 G, 以存储边(楼栋->出口)及其距离/容量
    sList = []; tList = []; lenList=[]; capList=[];
    for e = 1:length(netEdges)
        sList   = [sList,   netEdges(e).StartNode];
        tList   = [tList,   netEdges(e).EndNode];
        lenList = [lenList, netEdges(e).Length];
        capList = [capList, netEdges(e).MaxCap];
    end
    G = digraph(sList,tList,lenList);
    G.Edges.Capacity = capList';

    %% ============ 2. 决策变量 x(i,j)=0/1 ============
    % 对每栋楼 i, 在 nExit 个出口 j 里只可选1 => sum_j x(i,j)=1
    nVars = nBuilding*nExit;
    lb = zeros(nVars,1);
    ub = ones(nVars,1);

    % 线性约束(等式): sum_j x(i,j)=1
    Aeq = [];
    beq = [];
    for iB=1:nBuilding
        row = zeros(1,nVars);
        for jE=1:nExit
            varIdx = (iB-1)*nExit + jE;
            row(varIdx) = 1;
        end
        Aeq = [Aeq; row];
    end
    beq = ones(nBuilding,1);

    % 如果需要强制某些楼栋->某出口, forceAssign 中指定
    for iF=1:length(forceAssign)
        bldgID  = forceAssign(iF).BuildingID;
        exitID  = forceAssign(iF).ExitID;
        fixVal  = 1.0;
        varIdxFix = (bldgID-1)*nExit + exitID;
        lb(varIdxFix)= fixVal; 
        ub(varIdxFix)= fixVal;
        otherExit = setdiff(1:nExit, exitID);
        for tmp=otherExit
            fixVar = (bldgID-1)*nExit + tmp;
            lb(fixVar)=0; ub(fixVar)=0;
        end
    end

    %% ============ 3. 给A,b设置"每个出口分配楼栋数"区间[3,8] ============
    A = [];
    b = [];

    L_vec = [3, 3, 3];   % 每个出口最少要 3 栋楼
    U_vec = [8, 8, 8];   % 每个出口最多 8 栋楼

    for jE = 1:nExit
        % sum_i x(i,jE) <= U_vec(jE)
        rowMax = zeros(1,nVars);
        for iB=1:nBuilding
            varIdx= (iB-1)*nExit + jE;
            rowMax(varIdx)=1;
        end
        A= [A; rowMax];
        b= [b; U_vec(jE)];

        % sum_i x(i,jE) >= L_vec(jE) => -sum_i x(i,jE) <= -L_vec(jE)
        rowMin = -rowMax;
        A= [A; rowMin];
        b= [b; -L_vec(jE)];
    end

    %% ============ 4. 调用多目标进化算法 (NSGA-II) ============
    global BUILDINGS EXITS GRAPHobj alpha_ beta_ vMax_ nB nE
    BUILDINGS = buildingData;
    EXITS     = exitData;
    GRAPHobj  = G;
    alpha_    = alpha;
    beta_     = beta;
    vMax_     = vMax;
    nB        = nBuilding;
    nE        = nExit;

    options = optimoptions('gamultiobj',...
        'PopulationSize',80,...
        'MaxGenerations',80,...
        'CrossoverFraction',0.8,...
        'ParetoFraction',0.3,...
        'MutationFcn',@mutationadaptfeasible,...
        'PlotFcn',[],...
        'Display','iter'...
        );

    % 将 A,b,Aeq,beq,lb,ub, 和 @myNonlcon 一并传入
    [xSol,fval,exitflag,output,population,score] = ...
        gamultiobj(@myObjective, nVars, A,b, Aeq,beq, lb,ub, @myNonlcon, options);

    %% ============ 5. 输出结果并可视化 ============
    fprintf('\n=== NSGA-II 优化结束, exitflag=%d ===\n', exitflag);
    disp(output);

    disp('--- 部分Pareto解 (F1,F2) ---');
    for i=1:size(xSol,1)
        fprintf('解#%d => F1=%.2f, F2=%.2f\n', i, fval(i,1), fval(i,2));
    end

    %% ============ 6. 加权理想点法(WIPM)筛选解 ============
    F1min = min(fval(:,1));
    F2min = min(fval(:,2));
    w1=0.7; w2=0.3; z=2;

    gvals= zeros(size(xSol,1),1);
    for i=1:size(xSol,1)
        part1= (fval(i,1)-F1min)/(F1min+eps);
        part2= (fval(i,2)-F2min)/(F2min+eps);
        gvals(i)= w1*(part1^z) + w2*(part2^z);
    end
    [~, idxBest]= min(gvals);
    bestF1= fval(idxBest,1);
    bestF2= fval(idxBest,2);

    fprintf('\n----------------------------------------\n');
    fprintf('WIPM 最优解 idx=%d: F1=%.2f, F2=%.2f\n', idxBest,bestF1,bestF2);
    fprintf('决策方案:\n');
    decodeSolution(xSol(idxBest,:));
end


%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% 配置楼栋 / 出口 / 网络信息
function [buildingData, exitData, netEdges, vMax, alpha, beta, forceAssign] = SetupParameters()
    
    buildingFloors = [10,17,15,15,17,17,17,17,17,17,10,17,17,17,15,16,17,18,19];
    buildingPeoplePflr = [16,22,11,11,11,22,11,10,22,11,16,11,11,11,10,11,12,13,14];

    nB= length(buildingFloors);
    for i=1:nB
        buildingData(i).Name= sprintf('%d#', i);
        buildingData(i).Floors= buildingFloors(i);
        buildingData(i).PeopleFloor= buildingPeoplePflr(i);
        buildingData(i).TotalPeople= buildingFloors(i)* buildingPeoplePflr(i);
    end

    exitData(1).Name='East';  exitData(1).Width=9;   exitData(1).Capacity=900;
    exitData(2).Name='North'; exitData(2).Width=4;   exitData(2).Capacity=400;
    exitData(3).Name='South'; exitData(3).Width=3;   exitData(3).Capacity=300;

    vMax=1.19; alpha=0.5; beta=0.5;

    % distMatrix(19x3),自行设定,此处仅随意:
    distMatrix= [
        50,170,180; 140,60,160; 130,150,80; 90,110,115; 55,180,160;
        160,50,145;125,140,70; 95,120,105; 60,200,150; 155,70,140;
        120,130,65;110,115,110;58,195,140;180,65,120;115,145,60;
        65,190,150;170,55,140;100,120,80;90,110,100
    ];

    netEdges= struct([]);
    for iB=1:nB
        for jE=1:3
            idx= (iB-1)*3 + jE;
            netEdges(idx).StartNode= iB;
            netEdges(idx).EndNode  = nB+jE;
            netEdges(idx).Length   = distMatrix(iB,jE);
            capEdge= min(1.6*100, exitData(jE).Capacity);
            netEdges(idx).MaxCap= capEdge;
        end
    end

    forceAssign= struct([]); 
end

%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% 目标函数 (F1=最大时间, F2=累积拥挤度)
function f= myObjective(x)
    global BUILDINGS EXITS GRAPHobj alpha_ beta_ vMax_ nB nE

    exitLoad= zeros(nE,1);
    for iB=1:nB
        Ni= BUILDINGS(iB).TotalPeople;
        for jE=1:nE
            varIdx= (iB-1)*nE + jE;
            if x(varIdx)>0.5
                exitLoad(jE)= exitLoad(jE)+ Ni;
            end
        end
    end

    pVec= exitLoad ./ [EXITS.Capacity]';
    vj= vMax_* ones(nE,1);
    idxOver= (pVec>0.5);
    vj(idxOver)= vMax_.* exp(- alpha_.* pVec(idxOver));

    buildingTime= zeros(nB,1);
    for iB=1:nB
        for jE=1:nE
            varIdx= (iB-1)*nE + jE;
            if x(varIdx)>0.5
                edIdx= find(GRAPHobj.Edges.EndNodes(:,1)== iB & ...
                            GRAPHobj.Edges.EndNodes(:,2)== nB+jE);
                dist_ij=9999;
                if ~isempty(edIdx)
                    dist_ij= GRAPHobj.Edges.Weight(edIdx(1));
                end
                buildingTime(iB)= dist_ij / vj(jE);
                break;
            end
        end
    end

    F1= max(buildingTime);

    f_j= zeros(nE,1);
    idxCongest= (pVec>=0.5);
    f_j(idxCongest)= exp(beta_.* pVec(idxCongest));
    F2= sum(f_j);

    f= [F1,F2];
end

%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% 非线性约束： 0/1 + 不得超容量
function [c,ceq] = myNonlcon(x)
    global nB nE BUILDINGS EXITS

    ceq= [];
    for iB=1:nB
        for jE=1:nE
            varIdx= (iB-1)*nE + jE;
            xk= x(varIdx);
            ceq= [ceq; xk*(1-xk)];
        end
    end

    exitLoad= zeros(nE,1);
    for iB=1:nB
        for jE=1:nE
            varIdx= (iB-1)*nE + jE;
            if x(varIdx)>0.5
                exitLoad(jE)= exitLoad(jE)+ BUILDINGS(iB).TotalPeople;
            end
        end
    end
    c= exitLoad - [EXITS.Capacity]';
end

%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% 解码
function decodeSolution(x)
    global BUILDINGS EXITS nB nE

    for iB=1:nB
        nameB= BUILDINGS(iB).Name;
        Ni= BUILDINGS(iB).TotalPeople;
        chosenExit= '(none)';
        for jE=1:nE
            varIdx= (iB-1)*nE + jE;
            if x(varIdx)>0.5
                chosenExit= EXITS(jE).Name;
                break;
            end
        end
        fprintf('%s => %s (%.1f人)\n', nameB, chosenExit, Ni);
    end
end
