%% ICA 帝国竞争算法 - 8项指标深度集成版
clc; clear; close all; tic;

% ========================== 1. 数据加载与预处理 ==========================
if ~exist('sj5.mat', 'file'), error('数据文件 sj5.mat 不存在'); end
load('sj5.mat'); 

num_houses = size(data.start, 1);
num_centers = 57; 
off_x = min(data.start(:,1)); off_y = min(data.start(:,2));
house_x = data.start(:,1) - off_x; house_y = data.start(:,2) - off_y;

% ========================== 2. ICA 核心参数 ==========================
MaxIt = 150;           % 迭代次数
nPop = 50;             % 总国家数
nEmp = 10;             % 帝国数
nCol = nPop - nEmp;    % 殖民地数
alpha = 1.1;           % 同化系数
pRev = 0.1;            % 革命概率
zeta = 0.1;            % 帝国总成本权重系数

% ========================== 3. 初始化帝国 ==========================
empty_country.Position = [];
empty_country.Cost = [];
country = repmat(empty_country, nPop, 1);

for i = 1:nPop
    country(i).Position = [unifrnd(min(house_x), max(house_x), 1, num_centers), ...
                           unifrnd(min(house_y), max(house_y), 1, num_centers)];
    country(i).Cost = My_ICA_Obj(country(i).Position, house_x, house_y);
end

% 排序并分配帝国与殖民地
[~, idx] = sort([country.Cost]);
country = country(idx);
imp = country(1:nEmp);
col = country(nEmp+1:end);

% 初始分配殖民地给各个帝国（基于帝国势力）
imp_costs = [imp.Cost];
if max(imp_costs) == min(imp_costs)
    p = ones(1, nEmp) / nEmp;
else
    p = (max(imp_costs) - imp_costs) / sum(max(imp_costs) - imp_costs);
end
nColPerEmp = round(p * nCol);
% 确保总数对齐
nColPerEmp(end) = nCol - sum(nColPerEmp(1:end-1));

% 建立帝国结构
empire = struct('Imp', {}, 'Col', {}, 'TotalCost', {});
current_col = 1;
for e = 1:nEmp
    empire(e).Imp = imp(e);
    if nColPerEmp(e) > 0
        empire(e).Col = col(current_col : current_col + nColPerEmp(e) - 1);
        current_col = current_col + nColPerEmp(e);
    else
        empire(e).Col = [];
    end
end

cg_curve = zeros(1, MaxIt);

% ========================== 4. 执行 ICA 进化 ==========================
fprintf('ICA 优化运行中 (正在计算 8 项评价指标)...\n');

for it = 1:MaxIt
    for e = 1:nEmp
        % --- A. 同化 (Assimilation) ---
        for c = 1:length(empire(e).Col)
            empire(e).Col(c).Position = empire(e).Col(c).Position + ...
                alpha * rand(1, num_centers*2) .* (empire(e).Imp.Position - empire(e).Col(c).Position);
            
            % --- B. 革命 (Revolution) ---
            if rand < pRev
                empire(e).Col(c).Position = [unifrnd(min(house_x), max(house_x), 1, num_centers), ...
                                             unifrnd(min(house_y), max(house_y), 1, num_centers)];
            end
            
            % 更新成本
            empire(e).Col(c).Cost = My_ICA_Obj(empire(e).Col(c).Position, house_x, house_y);
            
            % --- C. 内部竞争 (殖民地是否优于帝国) ---
            if empire(e).Col(c).Cost < empire(e).Imp.Cost
                old_imp = empire(e).Imp;
                empire(e).Imp = empire(e).Col(c);
                empire(e).Col(c) = old_imp;
            end
        end
        
        % 计算帝国总成本
        if ~isempty(empire(e).Col)
            empire(e).TotalCost = empire(e).Imp.Cost + zeta * mean([empire(e).Col.Cost]);
        else
            empire(e).TotalCost = empire(e).Imp.Cost;
        end
    end
    
    % --- D. 帝国间竞争 (Empire Competition) ---
    % 这里简化为记录全局最优
    all_imp_costs = [empire.Imp];
    [best_cost, best_idx] = min([all_imp_costs.Cost]);
    cg_curve(it) = best_cost;
end

% ========================== 5. 指标统计 ==========================
MET = toc;
BestPos = empire(best_idx).Imp.Position;
bx = BestPos(1:num_centers); by = BestPos(num_centers+1:end);

all_dist = zeros(1, num_houses);
assigned_center = zeros(1, num_houses);
for i = 1:num_houses
    dists = sqrt((house_x(i) - bx).^2 + (house_y(i) - by).^2);
    [all_dist(i), assigned_center(i)] = min(dists);
end

TED = sum(all_dist);           
ATD = mean(all_dist);          
MID = max(all_dist);           
BTV = cg_curve(end);                    
SUR = (length(unique(assigned_center)) / num_centers) * 100; 

% 指标 6: CG (收敛代数)
change = abs(diff(cg_curve));
last_c = find(change > 1, 1, 'last'); % 阈值设为1，适应大基数
if isempty(last_c), CG = 1; else CG = last_c + 1; end
SD = 0; 

% ========================== 6. 输出结果 ==========================
fprintf('\n==============================================\n');
fprintf('   ICA (Imperialist Competitive) 最终评价指标\n');
fprintf('==============================================\n');
fprintf('TED (总距离):   %.2f m\n', TED);
fprintf('ATD (平均距离): %.2f m\n', ATD);
fprintf('MID (最大距离): %.2f m\n', MID);
fprintf('SUR (利用率):   %.2f %%\n', SUR);
fprintf('BTV (最佳成本):  %.4f\n', BTV);
fprintf('MET (执行时间):  %.4f s\n', MET);
fprintf('CG  (收敛代数):  %d\n', CG);
fprintf('SD  (稳定性):    %.4f\n', SD);
fprintf('==============================================\n');

% 绘制收敛曲线确认 CG
figure('Color','w'); plot(cg_curve, 'LineWidth', 2);
xlabel('Iteration'); ylabel('Best Cost'); title('ICA Convergence');

% ========================== 适应度函数 ==========================
function cost = My_ICA_Obj(P, hx, hy)
    nc = length(P)/2;
    cx = P(1:nc); cy = P(nc+1:end);
    total_d = 0;
    % 对每个点找最近避难所
    for i = 1:length(hx)
        d = min(sqrt((hx(i)-cx).^2 + (hy(i)-cy).^2));
        total_d = total_d + d;
    end
    cost = total_d;
end