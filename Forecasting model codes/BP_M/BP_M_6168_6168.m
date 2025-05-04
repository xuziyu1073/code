clc;
clear all;
close all
warning off
addpath pathA

%% 导入数据
rng(0);
data = readtable('EVCSs.csv');
time_str = string(data{:,1});
time = datetime(time_str, 'InputFormat', 'yyyyMMdd');
variables = data{:,2:end};
variables = fillmissing(variables, 'constant', 0);

%% 多元序列 多输入
seq = 7; % 预测长度
split_date = datetime('2023-04-08') + days(7 - seq);
start_date = time(1);
end_date = split_date + days(seq);

P_train = variables(time >= start_date & time < split_date - days(seq), 1:end)';
P_test  = variables(time >= split_date - days(seq) & time < end_date - days(seq), 1:end)';

T_train = variables(time >= start_date + days(seq) & time < split_date, 1:end)';
T_test  = variables(time >= split_date & time < end_date, 1:end)';

%% -- 
% 原数据列数是 8424，取前 8400
len = size(P_train,2);
zushu = floor(len / 168);

p_train = P_train(:, 1:zushu*168);  
t_train = T_train(:, 1:zushu*168);
p_test  = P_test(:, 1:168);
t_test  = T_test(:, 1:168);

%% 数据归一化
mu_input = mean(p_train, 2);
sigma_input = std(p_train, 0, 2);
p_train = (p_train - mu_input) ./ sigma_input;
p_test  = (p_test  - mu_input) ./ sigma_input;

mu_output = mean(t_train, 2);
sigma_output = std(t_train, 0, 2);
t_train = (t_train - mu_output) ./ sigma_output;
t_test  = (t_test  - mu_output) ./ sigma_output;


%% -- 将数据转化为元胞数组形式 --
% p_train 和 t_train 转换为 6 个元胞，每个元胞包含一个 168xzushu 的矩阵
p_train_cell = mat2cell(p_train, 6, repmat(168, 1, zushu));  % 6 元胞，每个元胞大小为 168xzushu
t_train_cell = mat2cell(t_train, 6, repmat(168, 1, zushu));  % 6 元胞，每个元胞大小为 168xzushu

% p_test 和 t_test 转换为 6 个元胞，每个元胞包含一个 168x1 的矩阵
p_test_cell = mat2cell(p_test, 6, 168);  % 6 元胞，每个元胞大小为 168x1
t_test_cell = t_test;  % 大小为 168x1

%% BP 神经网络构建
% 创建一个前馈神经网络
net = fitnet(72);  % 隐藏层大小可以根据需要调整

% 设置训练参数
net.trainParam.epochs = 500;     % 迭代次数
net.trainParam.lr = 0.001;         % 学习率
net.trainParam.showCommandLine = false;  % 禁用命令行输出
net.performParam.regularization = 0.1;
%% 训练 BP 神经网络
startTime = datetime('now');
[net, tr] = train(net, p_train_cell, t_train_cell);
endTime = datetime('now');
elapsedTime = endTime - startTime; 

%% 预测
t_pred_cell = net(p_test_cell);  % 针对每个元胞分别进行预测

% 反归一化预测结果
t_pred_cell = cell2mat(t_pred_cell) .* sigma_output + mu_output;  % 反归一化

%% 绘图示例
output_folder = 'BP_output_images';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

for i = 1:6
    figure;
    plot(t_pred_cell(i, :), 'b', 'DisplayName', 'Forecast');
    hold on;
    plot(T_test(i, :), 'r', 'DisplayName', 'True');
    hold off;
    xlim([1, 168]);
    legend;
    title(['EV ' num2str(i)]);
    xlabel('Time(h)');
    ylabel('Charging Load(kW)');
    print(fullfile(output_folder, ['Figure_' num2str(i)]), '-dpng', '-r300');
    close;
end
disp('所有图像已保存到文件夹中。');

mae_row = mean(abs(t_pred_cell - T_test), 2);         % 6 x 1
rmse_row = sqrt(mean((t_pred_cell - T_test).^2, 2));  % 6 x 1

total_mae = mean(mae_row);
total_rmse = mean(rmse_row);

%% 存入表格
filename = 'T_test_values_BP_多变量.xlsx';

% 写入 t_pred_cell 到第 1 个工作表，从 A1 开始
writematrix(t_pred_cell, filename, 'Sheet', 1, 'Range', 'A1');

% 比如从 A8 开始（前 7 行留作空行或作为分隔）
startRowForTTest = size(t_pred_cell,1) + 2;  % 这里 +2 是预留一行空行
startRange = sprintf('A%d', startRowForTTest);
writematrix(T_test, filename, 'Sheet', 1, 'Range', startRange);

%--------------------------------------------------%
% 4. 将行 MAE、RMSE、整体 MAE、RMSE、运行时间保存到 “Metrics” 工作表

elapsedTimeInSeconds = seconds(elapsedTime);

% 为了方便写表，可以先构造一个 cell 数组
Metrics = cell(1 + 6 + 3, 3);  % 1 行表头 + 6 行(每一行的MAE/RMSE) + 3 行(整体MAE/RMSE/时间)

% 填充表头
Metrics{1,1} = 'Row Index';
Metrics{1,2} = 'MAE';
Metrics{1,3} = 'RMSE';

% 填充每一行的 MAE、RMSE
for i = 1:6
    Metrics{i+1,1} = i;
    Metrics{i+1,2} = mae_row(i);
    Metrics{i+1,3} = rmse_row(i);
end

% 在最后三行写入整体指标和时间
Metrics{8,1}  = 'Overall MAE';
Metrics{8,2}  = total_mae;
Metrics{9,1}  = 'Overall RMSE';
Metrics{9,2}  = total_rmse;
Metrics{10,1} = 'Elapsed Time (s)';
Metrics{10,2} = elapsedTimeInSeconds;

% 将该 cell 写入到 Excel 文件
writecell(Metrics, filename, 'Sheet', 'Metrics');

% 把网络单独保存成 .mat
save('netBP.mat','net','-v7');  
% 用 dir 读文件大小
S = dir('netBP.mat');
modelSizeMB = S.bytes/1024^2;
fprintf('Model size on disk: %.2f MB\n',modelSizeMB);
