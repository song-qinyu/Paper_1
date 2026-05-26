%% ========= 示例：楼栋-中间节点-出入口 的图可视化 =========

% --- 1. 定义节点分类 ---
buildingNodes   = [1, 2];    % 楼栋节点
middleNodes     = [3, 4];    % 中间节点 (仅用于连接)
entranceNodes   = [5, 6];    % 出入口节点

% 楼栋对应的人数
buildingPopulations = [50; 80]; 

% 出入口属性（宽度、容量）
entranceWidths   = [3; 2]; 
entranceCapacity = [100; 60]; 

% --- 2. 构建邻接矩阵(距离) ---
% 这里示例一共 6 个节点 (1,2: 楼栋; 3,4: 中间; 5,6: 出入口)
% 以下矩阵表示各节点间的距离，0 表示不直连，后续转换为 Inf。
distanceMatrix = [
    %   1   2   3   4   5   6
    0   10   5   0   0   0;  % 节点1(楼栋1)到...
    10   0   0   5   0   0;  % 节点2(楼栋2)
    5   0   0   10   0   0;  % 节点3(中间节点1)
    0   5   10   0   10  0;  % 节点4(中间节点2)
    0   0   0   10   0   5;  % 节点5(出入口1)
    0   0   0   0    5   0   % 节点6(出入口2)
];

% 将 0（表示不连通）改为 Inf，以便在图中处理
tempMatrix = distanceMatrix;
tempMatrix(tempMatrix == 0) = Inf;
for i = 1:size(tempMatrix,1)
    tempMatrix(i,i) = 0;   % 自身到自身距离 = 0
end

% 创建加权图 (使用距离作为边权重)
G = graph(tempMatrix, 'upper');  % 'upper' 表示只使用上三角(对称图)

% --- 3. 添加节点属性 ---
numNodes = numnodes(G);
nodeTable = table((1:numNodes)', 'VariableNames', {'ID'});

% 通用字段
nodeTable.Population = zeros(numNodes,1);  % 楼栋人口（默认 0）
nodeTable.Width      = zeros(numNodes,1);  % 出入口宽度（默认 0）
nodeTable.Capacity   = zeros(numNodes,1);  % 出入口容量（默认 0）
nodeTable.Type       = repmat("Unknown", numNodes, 1); % 节点类型

% 为“楼栋节点”赋值
for i = 1:length(buildingNodes)
    bNode = buildingNodes(i);
    nodeTable.Population(bNode) = buildingPopulations(i);
    nodeTable.Type(bNode) = "Building";
end

% 为“中间节点”赋值 (这里演示：它们没有特殊属性，仅类型标记)
for i = 1:length(middleNodes)
    mNode = middleNodes(i);
    nodeTable.Type(mNode) = "Intermediate";
end

% 为“出入口节点”赋值
for i = 1:length(entranceNodes)
    eNode = entranceNodes(i);
    nodeTable.Width(eNode)    = entranceWidths(i);
    nodeTable.Capacity(eNode) = entranceCapacity(i);
    nodeTable.Type(eNode)     = "Entrance";
end

% 将属性赋给图
G.Nodes = nodeTable;

% 可选：若要给 Edges 添加更多字段，直接对 G.Edges.字段名 = 值
% 例：G.Edges.CapEdge = randi([10,50],height(G.Edges),1);

% --- 4. 绘图 ---
figure;
p = plot(G, ...
    'Layout', 'force', ...           % 力导向布局
    'EdgeLabel', G.Edges.Weight, ... % 边上标注距离
    'NodeLabel', G.Nodes.ID ...      % 节点上显示编号
    );
title('楼栋-中间节点-出入口示例图');

% 为便于区分，分别高亮不同类型的节点
hold on;
highlight(p, buildingNodes, 'NodeColor', 'red');          % 楼栋用红色
highlight(p, middleNodes,   'NodeColor', 'magenta');      % 中间节点用洋红色
highlight(p, entranceNodes, 'NodeColor', 'blue');         % 出入口用蓝色
hold off;

% --- 5. 打印信息以查看 ---
disp('=== 节点属性表 ===');
disp(G.Nodes);
disp('=== 边属性表 ===');
disp(G.Edges);
