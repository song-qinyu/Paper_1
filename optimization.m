function path_opt = optimization(map,path)
l_p = length(path(:,1)) ;%获得路径第一列的长度
path_opt = [path(1,1),path(1,2)] ;
path_tem = [path(1,1),path(1,2)] ;
dis_tem = zeros(1,l_p) ;%生成1*l_p的0矩阵
for i = 2:l_p    %从第二个点开始
    nodes = [path(i,1),path(i,2)];%路径中的第i个点
    dis_tem(i) = pdist2(nodes,path_tem);%计算第i个点和起点的距离 行向量的欧式距离
        if dis_tem(i)> 0 || dis_tem(i)> dis_tem(i-1)     
           indx = i;
           path_new = [path(indx,1) path(indx,2)];
        end
    if ~checkpath(path_tem,path_new,map)
        path_tem = [path(indx-1,1) path(indx-1,2)]; 
        path_opt = [path_opt; path_tem];
    end
end

end