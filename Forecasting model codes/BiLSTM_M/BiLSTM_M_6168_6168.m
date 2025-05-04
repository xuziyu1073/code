
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

%% -- 舍弃多余列使能整除 168 --
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

%% -- 2) 将数据 reshape 成 3D --
% 数据的形状应该是 [6, 168, zushu]，其中 6 是变量数，168 是时间步长，zushu 是组数。
p_train_3d = reshape(p_train, [6, 168, zushu]);  % [6,168,zushu]
t_train_3d = reshape(t_train, [6, 168, zushu]);
p_test_3d  = reshape(p_test,  [6, 168, 1]);
t_test_3d  = reshape(t_test,  [6, 168, 1]);


%% -- 将数据 reshape 成 3D --
% 将数据重构为元胞数组，每个元胞包含一个训练样本
% 每个样本的格式是 [时间步数, 特征数]
p_train_cell = cell(zushu, 1);  % zushu组数据
t_train_cell = cell(zushu, 1);  % 对应的目标数据
for i = 1:zushu
    p_train_cell{i} = squeeze(p_train_3d(:, :, i));  % 每个训练样本 [168, 6]
    t_train_cell{i} = squeeze(t_train_3d(:, :, i));  % 每个目标样本 [168, 6]
end

% 同理，重构测试集数据为元胞数组
p_test_cell = cell(1, 1);
t_test_cell = cell(1, 1);
p_test_cell{1} = squeeze(p_test_3d(:, :, 1));  % 测试样本 [168, 6]
t_test_cell{1} = squeeze(t_test_3d(:, :, 1));  % 测试目标样本 [168, 6]

%% LSTM 网络构建
inputSize = 6;  % 每个时间步的特征数
outputSize = 6;  % 目标输出的特征数

numHiddenUnits = 72;  
dropoutRate = 0.1;  
LearnRate=0.001;  

layers = [
    sequenceInputLayer(inputSize)  % 输入层，每个时间步的特征数是 6
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence')  % BiLSTM 层，返回每个时间步的输出
    dropoutLayer(dropoutRate)  % 添加 Dropout 层，丢弃50%的神经元
    fullyConnectedLayer(outputSize)  % 全连接层，输出特征数是 6
    regressionLayer  % 回归层，用于回归任务
];

% 训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', LearnRate, ...  
    'MaxEpochs', 500, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

%% 训练 LSTM 网络
% 使用元胞数组作为输入
startTime = datetime('now');
netLSTM = trainNetwork(p_train_cell, t_train_cell, layers, options);
endTime = datetime('now');
elapsedTime = endTime - startTime; 
%% 预测
% 使用测试数据进行预测
t_pred_cell = predict(netLSTM, p_test_cell);

% 将预测结果反归一化
t_pred_2d = t_pred_cell{1} .* sigma_output + mu_output;

%% 保存结果到 CSV

mae_row = mean(abs(t_pred_2d - T_test), 2);         % 6 x 1
rmse_row = sqrt(mean((t_pred_2d - T_test).^2, 2));  % 6 x 1
total_mae = mean(mae_row)
total_rmse = mean(rmse_row)

%% 绘图示例
output_folder = 'BiLSTM_output_images';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

for i = 1:size(t_pred_2d, 1)
    figure;
    plot(t_pred_2d(i, :), 'b', 'DisplayName', 'Forecast');
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



%% 存入表格
filename = 'T_test_values_BiLSTM_M_多变量.xlsx';

% 写入 t_pred_cell 到第 1 个工作表，从 A1 开始
writematrix(t_pred_2d, filename, 'Sheet', 1, 'Range', 'A1');

% 比如从 A8 开始（前 7 行留作空行或作为分隔）
startRowForTTest = size(t_pred_2d,1) + 2;  % 这里 +2 是预留一行空行
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
save('netLSTM.mat','netLSTM','-v7');  
% 用 dir 读文件大小
S = dir('netLSTM.mat');
modelSizeMB = S.bytes/1024^2;
fprintf('Model size on disk: %.2f MB\n',modelSizeMB);

