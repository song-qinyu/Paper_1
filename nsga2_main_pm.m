function mainEvacuationWIPM
    clc; clear; close all;

    % ========== 1) 载入疏散网络数据 & 显示网络 ==========    
    dataFile = 'sj.mat';
    S = load(dataFile);
    showNetworkFromData(dataFile);  % 仅用来画图看下结构

    % 若缺 EdgeCapacity, 设默认
    if ~isfield(S, 'EdgeCapacity')
        disp('[提示] 数据中未找到 EdgeCapacity, 统一给每条边容量=300');
        n = size(S.AdjMatrix,1);
        S.EdgeCapacity = zeros(n);
        for i = 1:n
            for j = 1:n
                if S.AdjMatrix(i,j) > 0
                    S.EdgeCapacity(i,j) = 300;
                end
            end
        end
    end

    % ========== 2) 定义决策变量数量 (楼栋*2) ==========
    buildingMask = (string(S.Nodes.Type) == "Building");
    buildingNodes = find(buildingMask);
    B = length(buildingNodes);
    nVar = 2*B; % 2个决策变量(出口,路径) per 楼栋

    % ========== 3) 先做极端求解, 得到 F1min, F2min ==========
    % 3.1) 最小化F1 => w1=1, w2=0 => gamultiobj or ga for single objective?
    %    这里为了方便, 我们直接拿 "gamultiobj" 也行,
    %    但事实上, 只想最小F1, F2不计算. 
    %    简易做法: 写 evacF1_wrapper 只输出F1
    userData = initUserData(S);

    disp('=== 正在最小化 F1(忽略F2) 以获取 F1min ===');
    problemF1 = @(x) evacF1_wrapper(x, userData);  % 只返回 F1
    [xF1, F1min] = ga(problemF1, nVar, [],[],[],[], ...
        zeros(1,nVar), ones(1,nVar), [], ...
        gaoptimset('Display','iter','PopulationSize',30,'Generations',30));

    disp('=== 正在最小化 F2(忽略F1) 以获取 F2min ===');
    problemF2 = @(x) evacF2_wrapper(x, userData);  % 只返回 F2
    [xF2, F2min] = ga(problemF2, nVar, [],[],[],[], ...
        zeros(1,nVar), ones(1,nVar), [], ...
        gaoptimset('Display','iter','PopulationSize',30,'Generations',30));

    disp('得到 F1min, F2min = ');
    disp([F1min, F2min]);

    % ========== 4) 用 WIPM 的目标函数, 做单目标优化 ==========
    w1=0.7; w2=0.3;
    problemWIPM = @(x) wipm_wrapper(x, userData, F1min, F2min, w1, w2);

    % 调用 ga (或 fmincon, 视情况)
    disp('=== 用 WIPM 目标, 进行单目标 ga 优化 ===');
    [xBest, fValWipm] = ga(problemWIPM, nVar, [],[],[],[], ...
        zeros(1,nVar), ones(1,nVar), [], ...
        gaoptimset('Display','iter','PopulationSize',50,'Generations',50));

    % ========== 5) 解析 & 显示结果 ==========
    disp('=== WIPM 最优解 对应的单目标值 ===');
    disp(fValWipm);

    % 再计算该解实际的 [F1,F2]
    [Fvals,~,~] = evacObjFun(xBest, userData); 
    F1final = Fvals(1); F2final = Fvals(2);

    disp('对应 F1, F2 = ');
    disp([F1final, F2final]);

    % 把楼栋->出口+路径 也打印出来
    decodeInfo = decodeSolution(xBest, userData.S, userData.allPaths);
    disp('=== 最优解 对应的楼栋 -> 出口, 路径 ===');
    printDecodedSolution(decodeInfo, userData.S);
end


%% ========== 一些辅助函数 ==========

function userData = initUserData(S)
    userData.S        = S;       
    userData.alpha    = 0.5;     
    userData.beta     = 0.5;     
    userData.vMax     = 12;      
    userData.timeStep = 1;       
    userData.maxSimulationTime = 600;

    buildingMask = (string(S.Nodes.Type)=="Building");
    buildingNodes = find(buildingMask);

    % 路径库(仅示例:1条最短路)
    userData.allPaths = buildCandidatePaths(S, buildingNodes);
end

%% ========== evacF1_wrapper: 只输出F1 ==========
function f = evacF1_wrapper(x, userData)
    [vals, ~, ~] = evacObjFun(x, userData);
    % vals = [F1, F2]
    f = vals(1); % 只返回 F1

end

%% ========== evacF2_wrapper: 只输出F2 ==========
function f = evacF2_wrapper(x, userData)
    [vals, ~, ~] = evacObjFun(x, userData);
    f = vals(2); % 只返回 F2
end

%% ========== wipm_wrapper: 计算 WIPM 单目标值 ==========
function val = wipm_wrapper(x, userData, F1min, F2min, w1, w2)
    [vals, ~, ~] = evacObjFun(x, userData);
    F1x=vals(1); F2x=vals(2);

    % WIPM:
    term1 = (F1x - F1min)/ max(eps, F1min);   % 做个保护,避免 /0
    term2 = (F2x - F2min)/ max(eps, F2min);

    val = w1*(term1^2) + w2*(term2^2);
end

%% ========== evacObjFun (原多目标F1,F2计算) ==========
function [f, g, h] = evacObjFun(x, userData)
    S     = userData.S;
    alpha = userData.alpha;
    beta  = userData.beta;
    vMax  = userData.vMax;
    dt    = userData.timeStep;
    maxT  = userData.maxSimulationTime;

    decodeInfo = decodeSolution(x, S, userData.allPaths);
    [edgePeople, nodePeople] = initEvacState(S, decodeInfo);

    tNow=0;
    totalCongest=0;
    allEvacuated=false;

    while ~allEvacuated && tNow<maxT
        tNow = tNow + dt;
        [edgePeople, nodePeople, cStep] = updateEvacOneStep(...
            edgePeople, nodePeople, S.AdjMatrix, S.EdgeCapacity, ...
            alpha, beta, vMax, dt);

        totalCongest = totalCongest + cStep;
        allEvacuated = checkAllEvacuated(nodePeople, S.Nodes);
    end

    F1 = tNow;
    F2 = totalCongest;
    f  = [F1,F2];
    g=[];
    h=[];
end


%% ========== decodeSolution: (与现有相同)==========
function decodeInfo = decodeSolution(x, S, allPaths)
    buildingMask = (string(S.Nodes.Type)=="Building");
    buildingNodes = find(buildingMask);
    entranceMask  = (string(S.Nodes.Type)=="Entrance");
    entranceNodes = find(entranceMask);

    B = length(buildingNodes);
    E = length(entranceNodes);

    decodeInfo.build2exit = zeros(B,1);
    decodeInfo.buildPaths = cell(B,1);

    for i=1:B
        varExit = x(2*i-1);
        varPath = x(2*i);

        if E<1
            decodeInfo.build2exit(i)=NaN;
            decodeInfo.buildPaths{i}=[buildingNodes(i)];
            continue;
        end

        step1 = 1/E;
        exitIdx = ceil(varExit/step1);
        exitIdx = max(exitIdx,1);
        exitIdx = min(exitIdx,E);
        exitID = entranceNodes(exitIdx);
        decodeInfo.build2exit(i)=exitID;

        candSet=allPaths{i, exitIdx};
        if isempty(candSet)
            decodeInfo.buildPaths{i}=[buildingNodes(i)];
            continue;
        end
        nCand=length(candSet);
        step2=1/nCand;
        pathIdx=ceil(varPath/step2);
        pathIdx=max(pathIdx,1);
        pathIdx=min(pathIdx,nCand);

        decodeInfo.buildPaths{i}=candSet{pathIdx}; 
    end
end

%% ========== initEvacState: (与现有相同) ==========
function [edgePeople, nodePeople] = initEvacState(S, decodeInfo)
    n=size(S.AdjMatrix,1);
    edgePeople=zeros(n);
    nodePeople=zeros(n,1);

    for i=1:height(S.Nodes)
        if string(S.Nodes.Type(i))=="Building"
            nodeID=S.Nodes.ID(i);
            pop=S.Nodes.Population(i);
            nodePeople(nodeID)=pop;
        end
    end
    decodePaths=decodeInfo.buildPaths;
    assignin('caller','decodePaths', decodePaths);
end

%% ========== updateEvacOneStep: (改成按比例移动,与理论一致) ==========
function [edgePeople, nodePeople, sumCongest] = updateEvacOneStep(...
    edgePeople, nodePeople, distMat, capMat, alpha, beta, vMax, dt)

    n = size(distMat,1);
    decodePaths=evalin('caller','decodePaths');
    sumCongest=0;

    for i=1:n
        for j=1:n
            if distMat(i,j)>0
                N_ij=edgePeople(i,j);
                C_ij=capMat(i,j);
                if C_ij<1e-9, C_ij=1; end
                p_ij=N_ij/C_ij;

                if p_ij<=0.5
                    v_ij=vMax;
                    f_ij=0;
                else
                    v_ij=vMax*exp(-alpha*p_ij);
                    f_ij=exp(beta*p_ij);
                end
                sumCongest = sumCongest + f_ij;

                L_ij=distMat(i,j);
                distanceCoverable=v_ij*dt;
                if distanceCoverable>=L_ij
                    arrival=N_ij; % 全部人到节点j
                else
                    fraction=distanceCoverable/L_ij;
                    arrival=floor(fraction*N_ij);
                end
                edgePeople(i,j)=N_ij - arrival;
                nodePeople(j)=nodePeople(j)+arrival;
            end
        end
    end

    % 把节点上的人流转到下一条边
    for k=1:length(decodePaths)
        pathN=decodePaths{k};
        for idx=1:length(pathN)-1
            curN=pathN(idx);
            nxtN=pathN(idx+1);
            popHere=nodePeople(curN);
            if popHere>0
                nodePeople(curN)=0;
                edgePeople(curN,nxtN)=edgePeople(curN,nxtN)+popHere;
            end
        end
    end
end

%% ========== checkAllEvacuated: (与现有相同) ==========
function yesno=checkAllEvacuated(nodePeople, nodeTable)
    bMask=(string(nodeTable.Type)=="Building");
    bID=nodeTable.ID(bMask);
    if any(nodePeople(bID)>0)
        yesno=false;
    else
        yesno=true;
    end
end

%% ========== buildCandidatePaths: 仅存最短路(演示) ==========
function allPaths = buildCandidatePaths(S, buildingNodes)
    G = graph(S.AdjMatrix);
    entranceNodes = find(string(S.Nodes.Type)=="Entrance");
    B = length(buildingNodes);
    E = length(entranceNodes);

    allPaths = cell(B, E);
    for b=1:B
        startN=buildingNodes(b);
        for e=1:E
            endN=entranceNodes(e);
            p=shortestpath(G,startN,endN);
            allPaths{b,e}={p}; 
        end
    end
end

%% ========== printDecodedSolution: 打印楼栋->出口->路径 ==========
function printDecodedSolution(decodeInfo, S)
    buildingMask = (string(S.Nodes.Type)=="Building");
    buildingNodes = find(buildingMask);

    for i=1:length(buildingNodes)
        bNode = buildingNodes(i);
        eID   = decodeInfo.build2exit(i);
        pathArr = decodeInfo.buildPaths{i};
        disp(['  楼栋节点 ',num2str(bNode),' -> 出口 ',num2str(eID), ...
            '，路径: [', num2str(pathArr), ']']);
    end
end
