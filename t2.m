function createManualGraphGUI
    % 创建图形窗口
    hFig  = figure('Name','手动建立节点和连线','NumberTitle','off','MenuBar','none',...
        'Position',[300 200 900 500]);
    
    % 创建坐标轴
    hAx   = axes('Parent',hFig,'Units','normalized','Position',[0.05 0.1 0.65 0.85]);
    axis(hAx, [0 100 0 100]);   % 坐标范围(0~100, 0~100)
    hold(hAx,'on');
    grid(hAx,'on');
    title(hAx,'请点击右侧按钮来添加节点或连线');
    
    % 在右侧放置文字标题
    uicontrol('Parent',hFig,'Style','text','String','操作区',...
        'Units','normalized','Position',[0.75 0.82 0.2 0.05],'FontSize',12,'FontWeight','bold');

    % “添加节点”按钮
    uicontrol('Parent',hFig,'Style','pushbutton','String','添加节点',...
        'Units','normalized','Position',[0.75 0.72 0.2 0.07],'FontSize',10,...
        'Callback',@cbAddNode);

    % “连线节点”按钮
    uicontrol('Parent',hFig,'Style','pushbutton','String','连线节点',...
        'Units','normalized','Position',[0.75 0.62 0.2 0.07],'FontSize',10,...
        'Callback',@cbConnectNodes);

    % “导出并保存数据”按钮（合并了导出和保存功能）
    uicontrol('Parent',hFig,'Style','pushbutton','String','导出并保存数据',...
        'Units','normalized','Position',[0.75 0.52 0.2 0.07],'FontSize',10,...
        'Callback',@cbExportAndSave);

    % 在 figure 的 guidata 中保存数据结构
    % -------------------------------------------------------
    data.nNodes      = 0;       % 当前节点数
    data.nodeCoords  = [];      % [nNodes x 2], 每个节点的(x,y)
    data.nodeType    = {};      % 节点类型: "Building"/"Intermediate"/"Entrance"
    data.population  = [];      % 楼栋人口
    data.width       = [];      % 出口宽度
    data.capacity    = [];      % 出口容量

    data.adjMatrix   = [];      % 邻接矩阵(0 表示无连接) -> 表示边的“距离”
    % 新增 edgeCapacity 矩阵(0 表示无连接) -> 表示边的“容量”
    data.edgeCapacity = [];

    guidata(hFig, data);

    %% ========== 回调函数：添加节点 ========== 
    function cbAddNode(~,~)
        % 第一步：让用户先选择节点类型
        choiceType = questdlg('选择节点类型','节点类型',...
            '楼栋','中间节点','出口','楼栋');
        if isempty(choiceType)
            return;  % 用户取消
        end
        
        % 根据节点类型，若需人口/宽度/容量，获取用户输入
        nodeTypeStr = "Intermediate";
        nodePop     = 0;
        nodeWidth   = 0;
        nodeCap     = 0;
        switch choiceType
            case '楼栋'
                nodeTypeStr = "Building";
                ansPop = inputdlg('请输入该楼栋人口:','楼栋属性',1,{'50'});
                if isempty(ansPop), return; end
                nodePop = str2double(ansPop{1});
                if isnan(nodePop), nodePop = 0; end
                
            case '中间节点'
                nodeTypeStr = "Intermediate";
                % 无需额外属性

            case '出口'
                nodeTypeStr = "Entrance";
                answer = inputdlg({'宽度:','容量:'}, '出口属性', 1, {'3','100'});
                if isempty(answer), return; end
                nodeWidth = str2double(answer{1});
                nodeCap   = str2double(answer{2});
                if isnan(nodeWidth), nodeWidth = 0; end
                if isnan(nodeCap),   nodeCap   = 0; end
        end
        
        % 第二步：让用户在坐标轴上点击位置
        [x,y,button] = ginput(1);
        if isempty(x) || isempty(y) || button<=0
            return; 
        end

        % 第三步：把节点加入数据结构
        s = guidata(hFig);
        newID = s.nNodes + 1;
        s.nNodes = newID;
        s.nodeCoords(newID,:) = [x,y];
        s.nodeType{newID}   = nodeTypeStr;
        s.population(newID) = nodePop;
        s.width(newID)      = nodeWidth;
        s.capacity(newID)   = nodeCap;

        % 扩展邻接矩阵
        oldSize = size(s.adjMatrix,1);
        newSize = newID;

        newAdj  = zeros(newSize);
        newAdj(1:oldSize,1:oldSize) = s.adjMatrix;
        s.adjMatrix = newAdj;

        % 同理，扩展 edgeCapacity 矩阵
        newCapMat = zeros(newSize);
        if ~isempty(s.edgeCapacity)
            newCapMat(1:oldSize,1:oldSize) = s.edgeCapacity;
        end
        s.edgeCapacity = newCapMat;

        % 在图上画出节点（不同类型用不同颜色）
        nodeColor = 'k';
        switch nodeTypeStr
            case "Building",     nodeColor = 'red';
            case "Intermediate", nodeColor = 'blue';
            case "Entrance",     nodeColor = 'magenta';
        end
        plot(hAx, x, y, 'o','MarkerSize',8,'MarkerFaceColor',nodeColor,'MarkerEdgeColor','k');
        text(hAx, x+1, y, sprintf('%d',newID),'FontSize',10,'Color','k');

        guidata(hFig,s);
    end

    %% ========== 回调函数：连线节点 ========== 
    function cbConnectNodes(~,~)
        s = guidata(hFig);
        n = s.nNodes;
        if n < 2
            warndlg('当前节点数不足2，无法连线！','提示');
            return;
        end

        % 在这里，我们同时输入 距离 & 边容量
        prompt   = {'节点1 ID:','节点2 ID:','距离(权重):','容量(EdgeCapacity):'};
        dlgTitle = '连线节点';
        defInput = {'1','2','10','300'};  % 默认
        answer   = inputdlg(prompt, dlgTitle, 1, defInput);
        if isempty(answer), return; end

        nodeA = str2double(answer{1});
        nodeB = str2double(answer{2});
        w     = str2double(answer{3});  % 距离(权重)
        c     = str2double(answer{4});  % 容量

        if any([nodeA,nodeB] < 1) || any([nodeA,nodeB] > n) || nodeA==nodeB || isnan(w) || isnan(c)
            warndlg('节点ID、距离或容量不合法。','提示');
            return;
        end

        % 更新邻接矩阵（无向图） => 距离
        s.adjMatrix(nodeA,nodeB) = w;
        s.adjMatrix(nodeB,nodeA) = w;

        % 更新容量矩阵（无向图） => 容量
        s.edgeCapacity(nodeA,nodeB) = c;
        s.edgeCapacity(nodeB,nodeA) = c;

        % 在图上画线并显示权重
        xA = s.nodeCoords(nodeA,1);  yA = s.nodeCoords(nodeA,2);
        xB = s.nodeCoords(nodeB,1);  yB = s.nodeCoords(nodeB,2);
        plot(hAx, [xA xB], [yA yB], 'k-');
        text(hAx, mean([xA xB]), mean([yA yB]), sprintf('%.2f',w),...
            'Color',[0.2 0.2 0.2],'FontSize',9);

        guidata(hFig,s);
    end

    %% ========== 回调函数：导出并保存数据 ========== 
    function cbExportAndSave(~,~)
        s = guidata(hFig);

        % 调用本地函数生成节点表和边列表(仅用于查看/打印)
        [Tnodes, edgeList] = buildNodeTableAndEdges(s);
        
        % 在命令行打印
        fprintf('\n============ 导出数据 ============\n');
        fprintf('当前节点数: %d\n', s.nNodes);
        disp('=== 节点表 ===');
        disp(Tnodes);
        disp('=== 邻接矩阵 (无向, 0表示无连接) ===');
        disp(s.adjMatrix);
        disp('=== 边列表 [nodeA nodeB weight] ===');
        disp(edgeList);

        % 若想查看容量矩阵，也可在命令行打印
        disp('=== 容量矩阵 (edgeCapacity) ===');
        disp(s.edgeCapacity);

        % 弹出文件保存对话框
        [filename, pathname] = uiputfile('*.mat','保存网络数据为');
        if isequal(filename,0)
            return; % 用户取消
        end
        fullpath = fullfile(pathname, filename);
        
        % 构造保存结构
        saveStruct.Nodes        = Tnodes;
        saveStruct.AdjMatrix    = s.adjMatrix;     % 距离矩阵
        saveStruct.EdgeList     = edgeList;        % 仅用于查看
        saveStruct.EdgeCapacity = s.edgeCapacity;  % 新增

        try
            save(fullpath, '-struct','saveStruct');
            msgbox(['已保存数据至: ' fullpath],'保存成功');
        catch ME
            errordlg(['保存失败: ' ME.message],'错误');
        end
    end

end  % 主函数结束

%% ========== 本地函数：构造节点表 & 边列表 ==========
function [Tnodes, edgeList] = buildNodeTableAndEdges(s)
    n = s.nNodes;
    if n==0
        Tnodes = table;
        edgeList = [];
        return;
    end

    % 对各属性做强制补齐或截断，确保长度一致
    if size(s.nodeCoords,1) < n
        s.nodeCoords = [s.nodeCoords; NaN(n - size(s.nodeCoords,1), 2)];
    elseif size(s.nodeCoords,1) > n
        s.nodeCoords = s.nodeCoords(1:n,:);
    end

    % 补齐或截断节点类型（cell数组）
    nodeTypeCell = fixCellArrayLength(s.nodeType, n, 'Intermediate');
    
    % 补齐其他数值向量
    popVec = fixVectorLength(s.population, n, 0);
    wVec   = fixVectorLength(s.width, n, 0);
    cVec   = fixVectorLength(s.capacity, n, 0);
    
    nodeID = (1:n)';
    xCoord = s.nodeCoords(:,1);
    yCoord = s.nodeCoords(:,2);
    
    varNames = {'ID','Type','X','Y','Population','Width','Capacity'};
    Tnodes = table(nodeID, nodeTypeCell(:), xCoord, yCoord, popVec(:), wVec(:), cVec(:), ...
                   'VariableNames', varNames);

    % 构造边列表（无向图，仅记录一次边）
    edgeList = [];
    if size(s.adjMatrix,1) < n || size(s.adjMatrix,2) < n
        bigger = zeros(n);
        bigger(1:size(s.adjMatrix,1), 1:size(s.adjMatrix,2)) = s.adjMatrix;
        s.adjMatrix = bigger;
    elseif size(s.adjMatrix,1) > n || size(s.adjMatrix,2) > n
        s.adjMatrix = s.adjMatrix(1:n, 1:n);
    end
    for i = 1:n
        for j = i+1:n
            w = s.adjMatrix(i,j);
            if w ~= 0
                edgeList = [edgeList; i, j, w];
            end
        end
    end
end

%% ========== fixVectorLength ==========
function vecOut = fixVectorLength(vecIn, desiredLen, fillVal)
    if length(vecIn) < desiredLen
        vecOut = [vecIn(:); repmat(fillVal, desiredLen - length(vecIn), 1)];
    else
        vecOut = vecIn(1:desiredLen);
    end
end

%% ========== fixCellArrayLength ==========
function out = fixCellArrayLength(cellArray, desiredLen, fillVal)
    if numel(cellArray) < desiredLen
        out = [cellArray, repmat({fillVal}, 1, desiredLen - numel(cellArray))];
    else
        out = cellArray(1:desiredLen);
    end
end
