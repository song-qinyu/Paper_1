function D = CalculateDist(X)
%CALCULATEDIST 三维坐标计算距离
num = size(X,1);
D = zeros(num);
for i = 1:num
    for j = 1:num
        D(i,j) = sqrt((X(i,1)-X(j,1))^2+...
                      (X(i,2)-X(j,2))^2);
    end
end
end