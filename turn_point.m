function [turning_points]=turn_point(A)
% 初始化拐点计数器
  angle_sum=[];%累积拐点角度集合
turning_points = 0;
% A=point';
% 设置夹角阈值，可以根据具体情况调整
angle_threshold = 20; % 

% 计算路径的拐点个数
for i = 2:size(A, 1)-1
    % 获取连续三个点的坐标
    p1 = A(i-1, :);
    p2 = A(i, :);
    p3 = A(i+1, :);
    
    % 计算形成的向量
    v1 = p2 - p1;
    v2 = p3 - p2;
    
    % 计算夹角（弧度制）
    angle = atan2(v2(2), v2(1)) - atan2(v1(2), v1(1));
    
    % 将弧度转换为度数
    angle_deg = rad2deg(angle);
    
    % 如果夹角大于阈值，则认为是一个拐点
    if abs(angle_deg) > angle_threshold
        turning_points = turning_points + 1;
        angle_sum=[angle_sum;angle_deg];
    end
end