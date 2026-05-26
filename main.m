clear,clc,close all
%% 使用CEC2005测试
Func_name = 'F8'; %测试函数名，以F5为例
[lb,ub,dim,fobj] = Get_CEC2005_details(Func_name); %获取函数变量上下界，维度，目标函数句柄

pop = 50; %种群数量
maxIter = 100; %最大迭代次数
%求解
[Best_pos, Best_fitness, Iter_curve, History_pos, History_best] = GSA(pop, maxIter, lb, ub, dim,fobj);

%绘图
figure
subplot(1,2,1)
[x,y,f] = Plot_CEC2005(Func_name); %获取F5曲面数据（展示维度最多为2维）
surfc(x,y,f,'LineStyle','none');
colormap winter
title(Func_name);
subplot(1,2,2)
grid on;
plot(Iter_curve, 'r--', 'linewidth', 1.5)
title('GSA迭代曲线');
xlabel('迭代次数');
ylabel('适应度值');

