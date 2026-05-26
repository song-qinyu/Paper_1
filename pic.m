clc
clear all

load('gooddata.mat')

Finall=[];
for solID = 1:size(xSolutions,1)
    
    xVec = xSolutions(solID,:);
    decodeInfo = decodeSolution(xVec, userData.S, userData.allPaths);
    decodePaths=decodeInfo.buildPaths;
    outnum=[0,0,0];
    for k=1:19
        pathNodes = decodePaths{k};
        if pathNodes{1}(end)==1
            outnum(1)=outnum(1)+S.Nodes.Population(pathNodes{1}(1));
        elseif pathNodes{1}(end)==2
            outnum(2)=outnum(2)+S.Nodes.Population(pathNodes{1}(1));
        elseif pathNodes{1}(end)==3
            outnum(3)=outnum(3)+S.Nodes.Population(pathNodes{1}(1));
        end
    end
    Finall=[Finall;fSolutions(solID,:),outnum,sum(abs(outnum-mean(outnum)))/10];
end

solID=35;
xVec = xSolutions(solID,:);
decodeInfo = decodeSolution(xVec, userData.S, userData.allPaths);
printDecodedSolution(decodeInfo, userData.S);
showNetworkFromData(dataFile)
decodePaths=decodeInfo.buildPaths;
hold on
FF(solID,:)
for k=1:19
    pathNodes = decodePaths{k};
    plot([S.Nodes.X(pathNodes{1}(1)),S.Nodes.X(pathNodes{1}(end))],...
        [S.Nodes.Y(pathNodes{1}(1)),S.Nodes.Y(pathNodes{1}(end))],'r-');hold on
    %             for m=1:length(pathNodes{1})-1
    %             plot([S.Nodes.X(pathNodes{1}(m)),S.Nodes.X(pathNodes{1}(m+1))],...
    %                 [S.Nodes.Y(pathNodes{1}(m)),S.Nodes.Y(pathNodes{1}(m+1))],'r-');hold on
    %             end
end
titleText = sprintf(['路径为：',num2str(Finall(solID,1))  , '  拥挤度为：',num2str(Finall(solID,2))  , '\n三个出口疏散人数分别为：',...
    num2str(Finall(solID,3)),'人  ',num2str(Finall(solID,4)),'人  ',num2str(Finall(solID,5)),'人']);
title(titleText)
fprintf(['三个出口疏散人数分别为：',num2str(FF(solID,3)),'人  ',num2str(FF(solID,4)),'人  ',num2str(FF(solID,5)),'人  \n'])


for i=1:35
     disp(['方案',num2str(i)  ,'的路径为：',num2str(Finall(i,1))  , '  拥挤度为：',num2str(Finall(i,2))]);
end


for solID = 1:size(xSolutions,1)
    
    xVec = xSolutions(solID,:);
    decodeInfo = decodeSolution(xVec, userData.S, userData.allPaths);
    decodePaths=decodeInfo.buildPaths;
    outnum=[0,0,0];
    for k=1:19
        pathNodes = decodePaths{k};
        if pathNodes{1}(end)==46
            outnum(1)=outnum(1)+S.Nodes.Population(pathNodes{1}(1));
        elseif pathNodes{1}(end)==47
            outnum(2)=outnum(2)+S.Nodes.Population(pathNodes{1}(1));
        elseif pathNodes{1}(end)==48
            outnum(3)=outnum(3)+S.Nodes.Population(pathNodes{1}(1));
        end
    end
    Finall=[Finall;fSolutions(solID,:),outnum,sum(abs(outnum-mean(outnum)))/10];
end



%% 打印解码结果 (楼栋->出口+路径)
function printDecodedSolution(decodeInfo, S)
% decodeInfo.build2exit(i) = 出口ID
% decodeInfo.buildPaths{i} = 节点序列
buildingMask = (string(S.Nodes.Type)=="Building");
buildingNodes = find(buildingMask);

for i=1:length(buildingNodes)
    bNode = buildingNodes(i);
    bID = bNode;  % node ID
    eID = decodeInfo.build2exit(i); % 出口节点ID
    pathArr = decodeInfo.buildPaths{i};
    disp(['  楼栋节点 ',num2str(bID),' -> 出口 ',num2str(eID), ...
        '，路径: [', num2str(pathArr{1}), ']']);
end
end

%% ========== decodeSolution ==========
function decodeInfo = decodeSolution(x, S, allPaths)
% x(2i-1)=>出口, x(2i)=>路径索引
buildingMask = (string(S.Nodes.Type)=="Building");
buildingNodes = find(buildingMask);
entranceMask  = (string(S.Nodes.Type)=="Entrance");
entranceNodes = find(entranceMask);

B = length(buildingNodes);
E = length(entranceNodes);

decodeInfo.build2exit = zeros(B,1);
decodeInfo.buildPaths = cell(B,1);

for i=1:B
    varExit = x(2*i-1);  % [0,1], 映射到 e in [1..E]
    varPath = x(2*i);    % [0,1], 映射到 path index
    if E<1
        decodeInfo.build2exit(i)=NaN;
        decodeInfo.buildPaths{i}=[buildingNodes(i)];
        continue;
    end
    
    % 1) 出口索引
    step1=1/E;
    exitIdx=ceil(varExit/step1);
    exitIdx=max(exitIdx,1);
    exitIdx=min(exitIdx,E);
    exitID = entranceNodes(exitIdx);
    decodeInfo.build2exit(i)=exitID;
    
    % 2) 在 allPaths{i, exitIdx}里选择具体路径
    candSet=allPaths{i, exitIdx};  % cell array of possible paths
    if isempty(candSet)
        decodeInfo.buildPaths{i}=[buildingNodes(i)];
        continue;
    end
    nCand=length(candSet);
    step2=1/nCand;
    pathIdx=ceil(varPath/step2);
    pathIdx=max(pathIdx,1);
    pathIdx=min(pathIdx,nCand);
    
    %         decodeInfo.buildPaths{i}=candSet{pathIdx};
    decodeInfo.buildPaths{i}=candSet;
end
end

%% ========== initEvacState ==========
function [edgePeople, nodePeople] = initEvacState(S, decodeInfo)
n=size(S.AdjMatrix,1);
edgePeople=zeros(n,n,19);
nodePeople=zeros(n,1,19);

% 把楼栋人口放到节点
for i=1:length(decodeInfo.buildPaths)
    %         height(S.Nodes)
    path=decodeInfo.buildPaths{i,1};
    
    nodeID=S.Nodes.ID(path{1}(1));
    pop=S.Nodes.Population(path{1}(1));
    nodePeople(nodeID,1,i)=pop;
end

% 把 decodePaths 存在 caller
decodePaths=decodeInfo.buildPaths;
assignin('caller','decodePaths', decodePaths);
end

%% ========== updateEvacOneStep ==========
function [edgePeople, nodePeople, sumCongest] = updateEvacOneStep(...
    edgePeople, nodePeople, distMat, capMat, alpha, beta, vMax, dt)

n = size(distMat,1);

% 取出 decodePaths (各楼栋的具体路径) 以备后续节点->边移动
decodePaths = evalin('caller','decodePaths');

sumCongest = 0;

alledgePeople=sum(edgePeople,3);
% ==============================
% 1) 先处理“边上人员”的流动 + 计算拥挤度
% ==============================
for i = 1:n
    for j = 1:n
        if distMat(i,j) > 0
            % ------ A. 获取信息 ------
            N_ij = alledgePeople(i,j);       % 此时边(i->j)上人数
            C_ij = capMat(i,j);           % 此边容量
            if C_ij < 1e-9, C_ij = 1; end  % 防止除0
            p_ij = N_ij / C_ij;           % 人员密度
            
            % ------ B. 根据拥挤度决定速度 + 拥挤度 ------
            if p_ij <= 0.5
                v_ij = vMax;              % 不拥挤, 自由速度
                f_ij = 0;                % 拥挤度=0
            else
                v_ij = vMax * exp(-alpha*p_ij);   % 拥挤衰减
                f_ij = exp(beta * p_ij);          % 拥挤度
            end
            sumCongest = sumCongest + f_ij;   % 累积进F2
            
            % ------ C. 根据速度+时间步, 计算能移动多少人 ------
            L_ij = distMat(i,j);             % 边长
            distanceCoverable = v_ij * dt;   % 本时间步能走多远
            if distanceCoverable >= L_ij
                % 整条边都能走完 => 所有人抵达节点 j
                arrival = N_ij;
                fraction=1;
            else
                % 按比例算
                fraction = distanceCoverable / L_ij;  % ∈ (0,1)
                %                     if N_ij==1
                %                         arrival  = max(floor(fraction * N_ij),1);    % 向下取整
                %                     else
                %                         arrival  = floor(fraction * N_ij);    % 向下取整
                %                     end
            end
            
            % ------ D. 更新该边&目标节点的人数 ------
            for k=1:19
                if edgePeople(i,j,k)==1
                    arrival  = max(floor(fraction * edgePeople(i,j,k)),1);    % 向下取整
                    edgePeople(i,j,k) = edgePeople(i,j,k) - arrival;
                    nodePeople(j,1,k)   = nodePeople(j,1,k) + arrival;
                elseif edgePeople(i,j,k)>1
                    arrival  = max(floor(fraction * edgePeople(i,j,k)),1);    % 向下取整
                    if arrival>edgePeople(i,j,k)
                        arrival=edgePeople(i,j,k);
                    end
                    edgePeople(i,j,k) = edgePeople(i,j,k) - arrival;
                    nodePeople(j,1,k)   = nodePeople(j,1,k) + arrival;
                    
                end
            end
        end
    end
end

% ==============================
% 2) 再让节点上的人进入下一个边(路径移动)
% ==============================
for k = 1:length(decodePaths)
    pathNodes = decodePaths{k};
    % 将在 pathNodes (curN->nxtN->...)中按顺序移动
    for idx = 1:length(pathNodes{1})-1
        curN = pathNodes{1}(idx);
        nxtN = pathNodes{1}(idx+1);
        
        popHere = nodePeople(curN,1,k);
        if popHere > 0
            % 该节点上所有人上边
            nodePeople(curN,1,k)   = 0;
            edgePeople(curN,nxtN,k) = edgePeople(curN,nxtN,k) + popHere;
        end
    end
end
end


%% ========== checkAllEvacuated ==========
function yesno=checkAllEvacuated(nodePeople, nodeTable,edgePeople)
bMask=(string(nodeTable.Type)=="Building");
bID=nodeTable.ID(bMask);
if any(nodePeople(bID)>0) || sum(sum(edgePeople))>0
    yesno=false;
else
    yesno=true;
end
end

