function showNetworkFromData(matFileName)
% showNetworkFromData - 从 GUI 保存的网络数据文件 (data1.mat) 读取并绘制网络
%
% 使用示例：
%   showNetworkFromData('data1.mat');

    if nargin<1
        error('请提供保存的 .mat 数据文件名，例如 showNetworkFromData(''data1.mat'')');
    end

    % 1) 加载数据
    S = load("sj.mat"); % 其中应有 S.Nodes, S.AdjMatrix, 以及(可选) S.EdgeCapacity, S.EdgeList
   
    % 简单检查
    if ~isfield(S,'Nodes') || ~isfield(S,'AdjMatrix')
        error('MAT 文件中缺少 Nodes 或 AdjMatrix，请检查数据文件。');
    end

    % 2) 构建图对象
    G = graph(S.AdjMatrix);

    % 若想在边上显示距离，可设置 G.Edges.Weight = ...
    % 默认 graph(AdjMatrix) 会把非零元素当做 Weight
    % (可以在 plot(...) 时通过 'EdgeLabel' 使用)

    % 3) 在 figure 中进行可视化
    figure('Name',['查看网络: ' matFileName],'NumberTitle','off');
    hold on;
    title('从数据文件读取的网络');

    % 若 Nodes 表中含 X, Y 两列，可用其定位节点
    if ismember('X', S.Nodes.Properties.VariableNames) && ...
       ismember('Y', S.Nodes.Properties.VariableNames)
        xCoords = S.Nodes.X;
        yCoords = S.Nodes.Y;
        h = plot(G, 'XData', xCoords, 'YData', yCoords, ...
                 'NodeLabel', S.Nodes.ID, ...
                 'EdgeLabel', G.Edges.Weight); 
    else
        % 若没有坐标，则用自动布局
        h = plot(G, ...
            'NodeLabel', S.Nodes.ID, ...
            'EdgeLabel', G.Edges.Weight);
    end

    % 4) 根据节点类型进行高亮/着色
    % 假设 S.Nodes.Type 存在并可能取值： "Building", "Entrance", "Intermediate"
    nodeTypes = string(S.Nodes.Type);
    bMask = (nodeTypes=="Building");
    eMask = (nodeTypes=="Entrance");
    mMask = (nodeTypes=="Intermediate") | (~bMask & ~eMask); % 其他认为中间节点

    bNodesID = S.Nodes.ID(bMask);
    eNodesID = S.Nodes.ID(eMask);
    mNodesID = S.Nodes.ID(mMask);

    % highlight 函数需要传递“节点在 plot 上的索引”
    % graph/plot 默认节点顺序对应 G.Nodes 中顺序，但我们要映射 ID -> index
    % 假设 ID 就是 1..n，则可直接 highlight ID
    % 如果 ID 不是从1开始依次递增，需要额外映射
    % 这里假设 ID = 1..n
    highlight(h, bNodesID, 'NodeColor','red');
    highlight(h, eNodesID, 'NodeColor','magenta');
    highlight(h, mNodesID, 'NodeColor','blue');

    % 5) 若想查看容量，可在命令窗口打印或在图中显示
    if isfield(S,'EdgeCapacity')
        disp('=== EdgeCapacity (容量矩阵) ===');
        disp(S.EdgeCapacity);
        % 可选择性把容量当做边标签： 
        %   set(h, 'EdgeLabel', arrayfun(@(x) num2str(x), S.EdgeCapacity(G.Edges.EndNodes), 'uni',0));
    end

    hold off;
    axis equal;
    grid on;

    % 在命令行打印节点表/边表(若需要)
    disp('=== Nodes 表 ===');
    disp(S.Nodes);
    if isfield(S,'EdgeList')
        disp('=== EdgeList ===');
        disp(S.EdgeList);
    end
    set(gca,'FontName','Microsoft YaHei');
    disp('完成可视化。');
end
