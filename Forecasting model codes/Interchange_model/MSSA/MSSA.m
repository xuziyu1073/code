clc; clear; close all;

%% 设置参数
M = 96;                    % 窗口长度 = 嵌入维度
N = 8760-168;              % 时间序列长度限制
tezheng_complete = M;     % 使用的主成分数量进行完整重构
limit_cumulative_contribution=0.6;
%% 从CSV文件中读取数据
filename = 'EVCSs.csv'; % CSV文件名称
start_date = 20220415;  % 起始日期
end_date = 20230407;    % 结束日期

% 读取表格数据
data = readtable(filename);

% 将date列转换为数字类型，便于区间筛选
% 假设 CSV 文件中有 'date' 列
if any(strcmp(data.Properties.VariableNames, 'date'))
    date_column = str2double(string(data{:, 'date'}));
else
    error('CSV 文件中缺少 "date" 列。');
end

% 根据日期区间进行过滤
filtered_rows = data(date_column >= start_date & date_column <= end_date, :);

% 提取EVCS1至EVCS6列的数据
EVCS_columns = {'EVCS1', 'EVCS2', 'EVCS3', 'EVCS4', 'EVCS5', 'EVCS6'};
if ~all(ismember(EVCS_columns, data.Properties.VariableNames))
    error('CSV 文件中缺少必要的 EVCS 列。');
end
X_original = filtered_rows{:, EVCS_columns};

% 如果提取的数据长度超过N，则限制为前N行
if size(X_original, 1) > N
    X_original = X_original(1:N, :);
end

% 检查数据是否足够
if size(X_original, 1) < M
    error('数据长度不足以构建轨迹矩阵。');
end

% 保存归一化前的均值和标准差（每个变量一个）
mu = mean(X_original, 1);            % 均值 (1 x 6)
sigma = std(X_original, 1, 1);       % 标准差 (1 x 6)

% 去除均值，并标准化为标准差1
X = (X_original - mu) ./ sigma;      % 标准化后的数据 (N x 6)

%% 构建完整的多变量轨迹矩阵 Y_full
% 轨迹矩阵的列数为 (轨迹数 * 变量数)
num_variables = size(X, 2); % 6
K = N - M + 1;              % 每个变量的轨迹数
Y_full = zeros(M, K * num_variables); % (M x (K * num_variables))

for m =1:num_variables
    Y_m = zeros(M, K); % 为每个变量创建 M x K 的轨迹矩阵
    for j=1:K
        Y_m(:,j) = X(j:j+M-1, m); % 提取窗口
    end
    Y_full(:, (m-1)*K +1 : m*K) = Y_m; % 拼接轨迹矩阵
end

%% 进行轨迹矩阵的特征值分解
% 计算协方差矩阵 C = Y_full * Y_full'
C = Y_full * Y_full';

% 特征值分解 C = U * Lambda * U'
[U, Lambda] = eig(C);

% 提取特征值并排序（降序）
Lambda_values = diag(Lambda);
[Lambda_values_sorted, idx] = sort(Lambda_values, 'descend');
U_sorted = U(:, idx);

% 计算所有值的总和
total_sum = sum(Lambda_values_sorted);
% 计算每个数相对于总和的贡献率
contribution_rate = Lambda_values_sorted / total_sum;
% 计算累计贡献率
cumulative_contribution = cumsum(contribution_rate);
disp('累计贡献率：');
disp(cumulative_contribution);

%% 特征断裂点选取
% 找到累积贡献值大于0.7的索引
valid_indices = find(cumulative_contribution > limit_cumulative_contribution);
rates = [];
for i = 1:length(valid_indices)-1
    rate= (Lambda_values_sorted(valid_indices(i)) - Lambda_values_sorted(valid_indices(i+1)))/Lambda_values_sorted(valid_indices(i));
    rates(i) = rate;
end
% 找到相邻特征变化率最大的前一个点，且同时满足了累计贡献率的要求的索引
[~, max_index] = max(rates);
result_index = min(valid_indices)-1+max_index;  %还原原始序列中的索引
tezheng_partial =result_index;        % 使用的主成分数量进行部分重构

%% 选择前 tezheng_complete 个主成分进行完整重构

num_components_complete = tezheng_complete;
if num_components_complete > size(U_sorted,2)
    error('num_components_complete 超过了 U 的列数。');
end
U_d_complete = U_sorted(:, 1:num_components_complete);     % (M x num_components_complete)

%% 选择前 tezheng_partial 个主成分进行部分重构

num_components_partial = tezheng_partial;
if num_components_partial > size(U_sorted,2)
    error('num_components_partial 超过了 U 的列数。');
end
U_d_partial = U_sorted(:, 1:num_components_partial);     % (M x num_components_partial)

%% 计算重构轨迹矩阵,步骤2的Y1...n    +    步骤3，分组，分组依据就是tezheng_partial = 4
% 完整重构：Y_recon_complete = U_d_complete * U_d_complete' * Y_full
Y_recon_complete = U_d_complete * (U_d_complete' * Y_full); % (M x K*num_variables)

% 部分重构：Y_recon_partial = U_d_partial * (U_d_partial' * Y_full)
Y_recon_partial = U_d_partial * (U_d_partial' * Y_full);   % (M x K*num_variables)

% 完整重构
reconstructed_complete = diagonal_average(Y_recon_complete, num_variables, M, K); % (N x 6)

% 部分重构
reconstructed_partial = diagonal_average(Y_recon_partial, num_variables, M, K);   % (N x 6)

%% 反归一化重构分量
RC_original_complete = reconstructed_complete .* sigma + mu; % 恢复到原始尺度
RC_original_partial = reconstructed_partial .* sigma + mu;   % 恢复到原始尺度

%% 比较重构后的时间序列与原始时间序列

% 反归一化原始时间序列
X_original_plot = X .* sigma + mu; % (N x 6)

%% 计算均方误差（MSE）以评估重构效果
mse_complete = mean((X_original_plot - RC_original_complete).^2, 1);
mse_partial = mean((X_original_plot - RC_original_partial).^2, 1);
disp('均方误差 (完整重构):');
disp(mse_complete);
disp('均方误差 (部分重构):');
disp(mse_partial);

%% 计算残差 bodong
bodong = X_original - RC_original_partial; % (N x 6)

% 提取EVCS1至EVCS6列的数据
EVCS_columns = {'EVCS1', 'EVCS2', 'EVCS3', 'EVCS4', 'EVCS5', 'EVCS6'};

%% 保存数据到 Excel 文件
% 提取 data 的第一列
data_first_column = filtered_rows{:, 1}; % 假设第一列在原始数据中是你需要的列

% 创建标题行
titles = [{'date'}, EVCS_columns(:)']; % 使用单元格数组创建标题行

% 将每个变量添加 data 的第一列
bodong_with_data = [data_first_column, bodong]; % 添加 data 列
RC_original_partial_with_data = [data_first_column, RC_original_partial]; % 添加 data 列
X_original_with_data = [data_first_column, X_original]; % 添加 data 列

% 将数据转换为单元格数组
bodong_with_data = num2cell(bodong_with_data);
RC_original_partial_with_data = num2cell(RC_original_partial_with_data);
X_original_with_data = num2cell(X_original_with_data);

% 将标题行添加到每个变量的最上方
bodong_with_data = [titles; bodong_with_data]; % 添加标题行
RC_original_partial_with_data = [titles; RC_original_partial_with_data]; % 添加标题行
X_original_with_data = [titles; X_original_with_data]; % 添加标题行

%%
excel_filename1 = sprintf('EVCSs_volatile_components.csv');
% 将每个数据集写入不同的工作表
writecell(bodong_with_data, excel_filename1);
disp(['数据已保存到文件: ' excel_filename1]);

%% 
excel_filename2 = sprintf('EVCSs_periodic_components.csv');
% 将每个数据集写入不同的工作表
writecell(RC_original_partial_with_data, excel_filename2);
disp(['数据已保存到文件: ' excel_filename2]);

%% 可视化奇异值分布
figure(1);
hold on;
circleSize = 4;     % 圆圈大小（对应特征值曲线）
starSize   = 4;    % 星号大小（对应累计贡献率曲线）
% 左侧坐标轴：特征值曲线（圆圈）
yyaxis left
plot(1:length(Lambda_values_sorted), Lambda_values_sorted, ...
     'o-', ...                         % 圆圈 + 连线
     'LineWidth', 1.5, ...
     'MarkerSize', circleSize, ...     % ← 圆圈大小
     'Color', [0.2, 0.6, 1]);          % 浅蓝色
xlabel('Number of eigenvalues', 'FontSize', 12, 'Color', 'k', ...
       'FontName', 'Times New Roman');
ylabel('Eigenvalue', 'FontSize', 12, 'Color', 'k', ...
       'FontName', 'Times New Roman');
ylim([0, max(Lambda_values_sorted)*1.1]);
set(gca, 'XColor', 'k', 'YColor', 'k', 'FontName', 'Times New Roman');

% 右侧坐标轴：累计贡献率曲线（星号）
yyaxis right
plot(1:length(cumulative_contribution), cumulative_contribution * 100, ...
     '*-', ...                         % 星号 + 连线
     'LineWidth', 1.5, ...
     'MarkerSize', starSize, ...       % ← 星号大小
     'Color', [1, 0.6, 0.2]);          % 浅橙色
ylabel('Cumulative contribution rate (%)', 'FontSize', 12, 'Color', 'k', ...
       'FontName', 'Times New Roman');
ylim([0, 100]);
set(gca, 'XColor', 'k', 'YColor', 'k', 'FontName', 'Times New Roman');

% 其他图形设置
box on
xlim([1, M]);
hold off;
% 以 300 DPI 保存为 PNG
exportgraphics(gcf, 'Figure1奇异值.png', 'Resolution', 300);

%% figure2
% 计算 y 轴的最大值和最小值
yMax_RC = max(RC_original_partial(:));
yMin_RC = min(RC_original_partial(:)); % 如果需要从最小值开始，可以使用 min(RC_original_partial(:))
% 获取 RC_original_partial 的行数，用于设置 x 轴限制
numRows_RC = size(RC_original_partial, 1);

% 创建一个新的图形窗口，编号为 2
figure(2);
clf; % 清空图形窗口
% 设置 tiledlayout，6 行 1 列，调整子图之间的间距
t_RC = tiledlayout(6,1, 'TileSpacing', 'compact', 'Padding', 'compact');
% 遍历每一列，创建子图
for i = 1:6
    nexttile;
    plot(RC_original_partial(:,i), 'LineWidth', 1, 'Color', [0.8,0.65,0.8510]); % 绘制第 i 列的数据
    ylim([yMin_RC yMax_RC]); % 设置 y 轴范围一致
    xlim([1 numRows_RC]); % 设置 x 轴范围为 RC_original_partial 的行数
    ylabel(['EVCS' num2str(i)], 'FontName', 'Times New Roman'); % 设置 y 轴标签为 EVCS1 到 EVCS6，字体为Times New Roman
    box on; % 设置坐标轴全封闭
    
    % 仅在最下方的子图显示 x 轴标签
    if i == 6
        xlabel('Time(h)', 'FontName', 'Times New Roman'); % 设置 x 轴标签，字体为Times New Roman
    else
        xticklabels([]); % 其他子图不显示 x 轴刻度标签
    end
    
    % 设置坐标轴字体为Times New Roman
    set(gca, 'FontName', 'Times New Roman');
end

% 调整图形窗口大小以适应标签和子图
set(gcf, 'Position', [10, 10, 400, 600]); % [left, bottom, width, height]
% 保存图像为 300 DPI 的 PNG 文件
filename = 'figure2-周期性部分.png';
exportgraphics(gcf, filename, 'Resolution', 300);
disp(['图像已成功保存为 ', filename, '，分辨率为 300 DPI。']);

%% figure3 
% 计算 y 轴的最大值和最小值
yMax = max(bodong(:));
yMin = min(bodong(:)); % 如果需要从最小值开始，可以使用 min(bodong(:))
% 获取 bodong 的行数，用于设置 x 轴限制
numRows = size(bodong, 1);

% 创建一个新的图形窗口，编号为 3
figure(3);
clf; % 清空图形窗口
% 设置 tiledlayout，6 行 1 列，调整子图之间的间距
t = tiledlayout(6,1, 'TileSpacing', 'compact', 'Padding', 'compact');
% 遍历每一列，创建子图
for i = 1:6
    nexttile;
    plot(bodong(:,i), 'LineWidth', 1, 'Color', [0.6686,0.8392,0.5275]); % 绘制第 i 列的数据
    ylim([yMin yMax]); % 设置 y 轴范围一致
    xlim([1 numRows]); % 设置 x 轴范围为 bodong 的行数
    ylabel(['EVCS' num2str(i)], 'FontName', 'Times New Roman'); % 设置 y 轴标签为 EVCS1 到 EVCS6，字体为Times New Roman
    box on; % 设置坐标轴全封闭
    
    % 仅在最下方的子图显示 x 轴标签
    if i == 6
        xlabel('Time(h)', 'FontName', 'Times New Roman'); % 设置 x 轴标签，字体为Times New Roman
    else
        xticklabels([]); % 其他子图不显示 x 轴刻度标签
    end
    
    % 设置坐标轴字体为Times New Roman
    set(gca, 'FontName', 'Times New Roman');
end

% 调整图形窗口大小以适应标签
set(gcf, 'Position', [10, 10, 400, 600]); % 调整图形窗口的位置和大小
% 保存图像为 300 DPI 的 PNG 文件
exportgraphics(gcf, 'figure3-波动性部分.png', 'Resolution', 300);
disp('图像已成功保存，分辨率为 300 DPI。');
