function P1 = coord(Shortest_Route,MM)
z = size(Shortest_Route,2);%最短路径个数
for Z = 1:z   %坐标转换
V = Shortest_Route(Z);
x0 = ceil(V/MM)-0.5;
y100 = mod(V,MM);
if y100==0
    y100 = MM;
end
y0 = MM+0.5-y100;
P1(1,Z) = x0;
P1(2,Z) = y0;
end
