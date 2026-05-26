function f = istouch(i, j, x, y, n)
    % 将节点转换为坐标
    [ix, iy] = nodeToCoord(i, n);
    [jx, jy] = nodeToCoord(j, n);

    % 检查直线是否穿过栅格 (x, y)
    f = isLineIntersectingGrid(ix, iy, jx, jy, x, y);
end

function [x, y] = nodeToCoord(node, n)
    x = mod(node - 1, n) + 1;
    y = floor((node - 1) / n) + 1;
end

function result = isLineIntersectingGrid(x1, y1, x2, y2, gx, gy)
    % 检查直线是否穿过栅格 (gx, gy)
    % 栅格的四个边
    left = gx - 0.5;
    right = gx + 0.5;
    bottom = gy - 0.5;
    top = gy + 0.5;

    % 检查直线是否与栅格的任何边相交
    result = isLineIntersectingLine(x1, y1, x2, y2, left, bottom, left, top) || ...
             isLineIntersectingLine(x1, y1, x2, y2, left, top, right, top) || ...
             isLineIntersectingLine(x1, y1, x2, y2, right, top, right, bottom) || ...
             isLineIntersectingLine(x1, y1, x2, y2, right, bottom, left, bottom);
end

function result = isLineIntersectingLine(x1, y1, x2, y2, x3, y3, x4, y4)
    % 检查两条线段是否相交
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
    if denom == 0
        result = false;
        return;
    end
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom;
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom;
    result = ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1;
end