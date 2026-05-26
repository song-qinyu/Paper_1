function new = adjust(G,road)
%G:地图
%road:路径
n = length(road);%原始路径长度
if(n <= 2)%如果路线顶点小于2，退出优化
    new = road;
    return;
end
new = [];%初始新路径为空
i = 1;%从第一点开始
while(i <= n)%开始循环
    new = [new,road(i)];%保存最新路径
    if(i == n)%当检测到最后一点
        break;%跳出循环
    end
    for j = n:-1:i+1%从最后一点往前检测
        if(isOK(G,road(i),road(j)) == 0)%检测原始路径的第i个和第j个是否连通
            continue;%如果不连通，继续检测j-1点
        else
            i = j;%如果连通，则把j点赋值给起始点i
            break;%跳出该循环
        end  
    end
end

