function [R_best,L_best,Shortest_Route,Shortest_Length] = tf_standard(D,initial,destination,dis,h,NC_max,m,MM,Lgrid)
%% 第一步：变量初始化
Rho=0.8; Q=100; Omega=5; Mu=1; Lambda=1.5;
Dir = [-MM-1,-1,MM-1,MM,MM+1,1,1-MM,-MM];
% Eta=1./D;%Eta为启发因子，这里设为距离的倒数
Tau=10*ones(MM^2,8);%Tau为信息素矩阵，初始化全为1，
%%
NC=1;                         %迭代计数器
R_best=zeros(NC_max,MM^2);    %各代最佳路线(行数为最大迭代次数NC_max，列数为走过栅格数量)
R_best_to_direct=zeros(NC_max,MM^2);%各代最佳路线（转移方向）
L_best=inf.*ones(NC_max,1);   %各代最佳路线的长度（inf:无穷大）
L_worst=zeros(NC_max,1);%各代最差路线长度
F_best=zeros(NC_max,1);     %各代最佳路线高度均方差
T_best=zeros(NC_max,1);     %各代最佳路线转弯次数
L_ave=zeros(NC_max,1);        %各代路线的平均长度
% inum = MM+(initial(1)/Lgrid-0.5)*MM-(initial(2)/Lgrid-0.5); %初始坐标转换为栅格标号
% dnum = MM+(destination(1)/Lgrid-0.5)*MM-(destination(2)/Lgrid-0.5); %终点坐标转换为栅格标号
inum = MM + (initial(1)-0.5)*MM - (initial(2)-0.5); %初始坐标转换为栅格标号
dnum = MM + (destination(1)-0.5)*MM - (destination(2)-0.5); %终点坐标转换为栅格标号
Tabu=zeros(m,MM^2);           %存储并记录路径的生成tabu:（停止，禁忌表）（m行矩阵）
to_direct=zeros(m,MM^2);         %存储并记录路径的转移方向过程（m行矩阵）
while NC<=NC_max              %停止条件之一：达到最大迭代次数
%% 第二步：m只蚂蚁按概率函数选择下一栅格
Alpha = 1;
Beta = 6;
%% 
   Tabu(:,1)=inum;     %将初始栅格加入禁忌表，所有蚂蚁都在起点出发
   for i=1:m
       j=2;       %栅格从第二个开始
       while Tabu(i,j-1)~=dnum%当当前蚂蚁所在栅格不是终点栅格时
            visited=Tabu(i,1:(j-1));      %已访问的栅格
            J=zeros(1,1);         %待访问的栅格
            N=J;        %待访问的栅格转移方向
            Pz=J;        %转移概率分布
            Phi=J;       %启发式信息概率分布，改进的
            Pll=J;
            Jc=1;       %循环下标，便于存储栅格
            Eta=J;
            for k=1:8   %利用循环求解待访问的栅格，如果第k个栅格不属于已访问的栅格，则其为待访问的栅格
                k1 = Dir(k)+visited(end);
                if D(visited(end),k)==inf
                    continue
                end
                if isempty(find(visited==k1, 1)) % if length(find(visited==k))==0,,,,,,,
                    J(Jc)=k1;% 含待访问栅格标号矩阵
                    N(Jc)=k; % 含待访问栅格转移标号矩阵
                    Jc=Jc+1;  %下标加1，便于下一步存储待访问的栅格
                end
            end
            if J==0        %死路的情况
                Tabu(i,:)=0;
                to_direct(i,:)=0;
                break
            end
            max_dis = max(dis(J));%改进的

%计算待访问栅格的转移概率分布和启发式信息概率分布
            for k=1:length(J)           %sum(J>0)表示待访问的栅格的个数
                Eta = [];
                Eta = 1/(D(visited(end),N(k)));
                Pz(k)=(Tau(visited(end),N(k))^Alpha)*(Eta^Beta);  %传统概率计算公式中的分子
            end
            Pz=Pz/(sum(Pz));               %传统转移概率分布：长度为待访问栅格个数
            P = Pz;
%             P = Pz;
%% 改进转移概率，轮盘赌
            %按概率原则选取下一个栅格
            Pcum=cumsum(P); %cumsum求累加和: cumsum([1 1 1])= 1 2 3，求累加的目的在于使Pcum的值总有大于rand的数
            Select=find(Pcum>=rand);    %当累积概率和大于给定的随机数，则选择个被加上的最后一个栅格作为即将访问的栅格
            if isempty(N)||isempty(Select)%蚂蚁死掉跳出循环
                break
            end
%% 
            to_direct(i,j-1) = N(Select(1));     %to_direct表示即将访问的栅格转移方向
            Tabu(i,j)=J(Select(1));          %将访问过的栅格加入禁忌表中
            j=j+1;         
        end
   end
    if NC>=2            %如果迭代次数大于等于2，则将上一次迭代的最佳路线存入Tabu的第一行中
        Tabu(1,:)=R_best(NC-1,:);
        to_direct(1,:)=R_best_to_direct(NC-1,:);
    end

%% 第三步：记录本次迭代最佳路线
    L=zeros(m,1);
    F=zeros(m,1);
    T=zeros(m,1);
    for i=1:m
            if Tabu(i,:)==0          %去掉死路的情况
               L(i)=inf; 
               continue
           end 
%            F(i)=std(h(Tabu(i,:)~=0));  %求走过路径的高度的均方差
           j=2;
           L(i)=Lgrid*D(Tabu(i,1),to_direct(i,1));
           while Tabu(i,j+1)~=0
              L(i)=L(i)+Lgrid*D(Tabu(i,j),to_direct(i,j));  %求路径距离
              T(i)=T(i)+~(~(to_direct(i,j)-to_direct(i,j-1))); %求转弯的次数
              j=j+1;
           end
    end
    L_sort = sort(L(:));%找路径排序
    L_sort1 = L_sort(L_sort~=inf);
    h9 = ceil(0.5*size(L_sort1,1));%确定h值
    L_best(NC)=min(L);              %最优路径为距离最短的路径
    if L_best(NC)==inf
%         error('没有通路');
        continue
    end
    L_worst(NC)=max(L(L~=inf));%各代最差路径长度
    pos=find(L==L_best(NC));         %找出最优路径对应的位置：即是哪只个蚂蚁
    R_best(NC,:)=Tabu(pos(1),:);       %确定最优路径对应的栅格顺序
    R_best_to_direct(NC,:)=to_direct(pos(1),:); %确定最优路径对应的栅格转移方向顺序
    F_best(NC) = F(pos(1));          %各代最优路线高度均方差
    T_best(NC) = T(pos(1));          %各代最优路线转弯次数
    L_ave(NC)=mean(L(L~=inf));              %求第k次迭代的平均距离(去掉死路的情况)
    disp 当前迭代次数
    NC = NC+1 

%% 第四步：更新信息素
    Delta_Tau=zeros(MM^2,8);           %Delta_Tau（i,j）表示所有的蚂蚁留在第i个栅格到相邻8个栅格路径上的信息素增量
    Delta_Tau_k=zeros(MM^2,8);
    for i=1:m
        for j=1:MM^2  %建立了完整路径后在释放信息素：蚁周系统Q/L
            if Tabu(i,j)==0||Tabu(i,j+1)==0%||或
                break
            else
             %Delta_Tau(Tabu(i,j),to_direct(i,j))=Delta_Tau(Tabu(i,j),to_direct(i,j))+Q/L(i);
                for k9 = 1:(h9-1)
                    Delta_Tau_k(Tabu(i,j),to_direct(i,j)) = Delta_Tau_k(Tabu(i,j),to_direct(i,j))+(h9-k9)*(1/L_sort1(k9));%对应文献1.2.4节
                 end
                    Delta_Tau(Tabu(i,j),to_direct(i,j)) = Delta_Tau(Tabu(i,j),to_direct(i,j))+(Delta_Tau_k(Tabu(i,j),to_direct(i,j))+h9*(1/L_sort1(1)));
            end
        end 
    end
    Tau=(1-Rho).*Tau+Delta_Tau;     %信息素更新公式
for i=1:MM^2%对Tau加限制条件，小于0的设为0.5
        for j=1:8
            if Tau(i,j)<0
                Tau(i,j) = 0.5;
            end
        end
 end
%% 第五步：禁忌表清零
    Tabu=zeros(m,MM^2);  %每迭代一次都将禁忌表清零
    to_direct=zeros(m,MM^2);  %转移方向矩阵清零
end
%% 第六步：输出结果
Pos=find(L_best==min(L_best));      %找到L_best中最小值所在的位置并赋给Pos
Shortest_Route=R_best(Pos(1),:);     %提取最短路径
Shortest_Route=Shortest_Route(Shortest_Route~=0);
Shortest_Length=L_best(Pos(1));     %提取最短路径的长度
end