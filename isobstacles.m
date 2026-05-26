function t = isobstacles(G,x,y)
%G:地图
%点（x,y）
%判断点（x,y）在地图G中是否是障碍物
n = length(G);
if G(n-y+0.5,x+0.5) == 1
    t = 1;%是障碍物
else
    t = 0;%不是障碍物
end

