function t = isOK(G, i, j)
    t = 1; % 默认连通
    n = size(G, 1); % 地图大小

    % 将节点转换为坐标
    [x1, y1] = nodeToCoord(i, n);
    [x2, y2] = nodeToCoord(j, n);

    % 使用 DDA 算法生成直线路径
    [x, y] = getLineGrids(x1, y1, x2, y2);

    % 检查路径上的所有栅格
    for k = 1:length(x)
        if x(k) < 1 || x(k) > n || y(k) < 1 || y(k) > n || G(y(k), x(k)) == 1
            t = 0; % 遇到障碍物或超出边界
            return;
        end
    end
end

function [x, y] = getLineGrids(x1, y1, x2, y2)
    % DDA 算法生成直线路径
    dx = x2 - x1;
    dy = y2 - y1;
    steps = max(abs(dx), abs(dy));
    x = round(linspace(x1, x2, steps + 1));
    y = round(linspace(y1, y2, steps + 1));
end

function [x, y] = nodeToCoord(node, n)
    % 节点编号转坐标
    y = ceil(node / n); % 行号（垂直方向）
    x = mod(node - 1, n) + 1; % 列号（水平方向）
end