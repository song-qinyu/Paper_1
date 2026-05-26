function [i_,j_] = change(i,j,n)
%把i和j点变为i在地图左边，j在地图右边
%n用于坐标变换
ix = mod(i,n) - 0.5;
if ix == -0.5
    ix = n - 0.5;
end
jx = mod(j,n) - 0.5;
if jx == -0.5
    jx = n - 0.5;
end
if ix <= jx
    i_ = i;
    j_ = j;
else
    i_ = j;
    j_ = i;
end