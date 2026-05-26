function path_new = reduce(path,obs)
path = path';obs = obs';
path_new = [path(1,1) path(1,2)]; % 当前点
P1 = [path(1,1) path(1,2)]; % 当前点

for i = 2:size(path,1)-1
    P2 = [path(i,1) path(i,2)]; % 下一点
    
    % 计算直线的方向向量
    v = P2 - P1;
    v_length = norm(v);
    v_hat = v / v_length; % 单位方向向量
    
    % 计算障碍物距离
    d_obs = inf;
    for i_obs = 1:size(obs,1)
        P = [obs(i_obs,1) - 0.5, obs(i_obs,2) - 0.5]; % 障碍物中心
        
        % 计算向量w = P - P1
        w = P - P1;
        
        % 计算投影长度
        proj_length = dot(w, v_hat);
        
        % 判断投影是否在线段上
        if proj_length < 0
            % 投影在P1之前，计算到P1的距离
            dist = norm(P - P1);
        elseif proj_length > v_length
            % 投影在P2之后，计算到P2的距离
            dist = norm(P - P2);
        else
            % 投影在线段上，计算垂直距离
            dist = norm(w - proj_length * v_hat);
        end
        
        if dist < d_obs
            d_obs = dist;
        end
    end
    
    if d_obs < 0.72 % 如穿过了障碍物
        path_new = [path_new; path(i-1,1) path(i-1,2)]; % 更新路径
        P1 = P2; % 更新当前点
    end
end

path_new = [path_new; path(end,1) path(end,2)];