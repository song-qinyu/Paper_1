load('sj5.mat');  % 必须包含 Finall、xSolutions、data

[~, idx] = sort(Finall(:,2));  % 疏散人数最均衡方案
xVec = xSolutions(idx(1), :);
xVec(xVec<0.01)=0.01;
X = ceil(xVec .* data.P);

% 初始化线条使用次数字典
road_use_count = containers.Map('KeyType','char','ValueType','double');

% 初始化图形
figure;
for i=1:length(data.road)
    temp = data.road{i};
    plot(temp(:,1), temp(:,2), 'k-'); hold on
end

% 已分配的直接画
for i=1:size(data.FID,1)
    a = [data.start(data.FID(i,1),1), data.binan(data.FID(i,2),1)];
    b = [data.start(data.FID(i,1),2), data.binan(data.FID(i,2),2)];
    plot(a,b,'b-'); hold on
end

% 可选路径部分（记录线段使用次数）
for i=1:length(X)
    s_id = data.DFenPei{i}(1);
    b_id = data.DFenPei{i}(X(i)+1);
    x_pair = [data.start(s_id,1), data.binan(b_id,1)];
    y_pair = [data.start(s_id,2), data.binan(b_id,2)];

    % 记录使用次数
    key = sprintf('%.2f-%.2f-%.2f-%.2f', x_pair(1), y_pair(1), x_pair(2), y_pair(2));
    if isKey(road_use_count, key)
        road_use_count(key) = road_use_count(key) + 1;
    else
        road_use_count(key) = 1;
    end
end

% 按照使用次数画线，线宽随次数变粗
keysList = keys(road_use_count);
for i=1:length(keysList)
    key = keysList{i};
    vals = sscanf(key, '%f-%f-%f-%f');
    x_pair = [vals(1), vals(3)];
    y_pair = [vals(2), vals(4)];
    count = road_use_count(key);
    plot(x_pair, y_pair, 'r-', 'LineWidth', 1 + count * 0.5); hold on
end

h1 = scatter(data.binan(:,1), data.binan(:,2), 18, 'g^', 'filled');
h2 = scatter(data.start(:,1), data.start(:,2), 8, 'ro', 'filled');
title('每个疏散点疏散人数最均衡方案（按通道使用次数加粗）');
legend([h1, h2], '避难点', '宅基地');
set(gca, 'FontName', 'Microsoft YaHei');
print(gcf, '-djpeg', '-r300', '最均衡方案-通道加粗.jpg');
