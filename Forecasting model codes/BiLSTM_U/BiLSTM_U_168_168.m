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

%% 选择这一列，这是单变量预测模型，需要手动依次选择列2-7,为了方便进行循环预测。
for i=2:7
variables = data{:,i};
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
% 数据的形状应该是 [1, 168, zushu]，其中 1 是变量数，168 是时间步长，zushu 是组数。
p_train_3d = reshape(p_train, [1, 168, zushu]);  % [1,168,zushu]
t_train_3d = reshape(t_train, [1, 168, zushu]);
p_test_3d  = reshape(p_test,  [1, 168, 1]);
t_test_3d  = reshape(t_test,  [1, 168, 1]);


%% -- 将数据 reshape 成 3D --
% 将数据重构为元胞数组，每个元胞包含一个训练样本
% 每个样本的格式是 [时间步数, 特征数]
p_train_cell = cell(zushu, 1);  % zushu组数据
t_train_cell = cell(zushu, 1);  % 对应的目标数据
for i = 1:zushu
    p_train_cell{i} = squeeze(p_train_3d(:, :, i));  % 每个训练样本 [168, 1]
    t_train_cell{i} = squeeze(t_train_3d(:, :, i));  % 每个目标样本 [168, 1]
end

% 同理，重构测试集数据为元胞数组
p_test_cell = cell(1, 1);
t_test_cell = cell(1, 1);
p_test_cell{1} = squeeze(p_test_3d(:, :, 1));  % 测试样本 [168, 1]
t_test_cell{1} = squeeze(t_test_3d(:, :, 1));  % 测试目标样本 [168, 1]

%% LSTM 网络构建
inputSize = 1;  % 1*168
numHiddenUnits1 = 72;  % 隐藏层的单元数
outputSize = 1;  % 1*168

layers = [
    sequenceInputLayer(inputSize)  % 输入层，每个时间步的特征数是 1
    bilstmLayer(numHiddenUnits1, 'OutputMode', 'sequence')  % LSTM 层，返回每个时间步的输出
    dropoutLayer(0.1)  
    fullyConnectedLayer(outputSize)  % 全连接层，输出特征数是 1
    regressionLayer  % 回归层，用于回归任务
];

% 训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...  
    'MaxEpochs', 500, ...
    'Verbose', 0);

%% 训练 LSTM 网络
% 使用元胞数组作为输入
startTime = datetime('now');
netLSTM = trainNetwork(p_train_cell, t_train_cell, layers, options);
endTime = datetime('now');
elapsedTime = endTime - startTime; 
elapsedTimeInSeconds = seconds(elapsedTime);
%% 预测
% 使用测试数据进行预测
t_pred_cell = predict(netLSTM, p_test_cell);

% 将预测结果反归一化
t_pred_2d = t_pred_cell{1} .* sigma_output + mu_output;

%%
mae = mean(abs(T_test - t_pred_2d));   % 计算 MAE
rmse = sqrt(mean((T_test - t_pred_2d).^2));  % 计算 RMSE
total_mae = mean(mae);
total_rmse = mean(mae);

disp(['Mean Absolute Error (MAE): ', num2str(mae)]);
disp(['Root Mean Squared Error (RMSE): ', num2str(rmse)]);

figure;
plot(T_test, 'b-', 'LineWidth', 1.5); hold on;
plot(t_pred_2d, 'r-', 'LineWidth', 1.5);
legend('真实值', '预测值', 'Location', 'best');
xlabel('样本索引');
ylabel('数值');
title('测试集真实值与预测值对比');
grid on;

% 保存为300 DPI的JPG图片
saveas(gcf, 'comparison_plot_LSTM.jpg', 'jpg');   % 保存为JPG格式
print('comparison_plot_LSTM', '-djpeg', '-r300');  % 以300 DPI保存

filename = 'T_test_values_BiLSTM_单变量.xlsx';

% 如果文件已经存在，则读取原有数据
if isfile(filename)
    % 读取现有数据
    [~, ~, existingData] = xlsread(filename);
    
    % 将新的 t_pred_2d 数据拼接到现有数据的右边
    newData = [existingData, num2cell(t_pred_2d')];  
else
    % 如果文件不存在，则创建新文件并将 t_pred_2d 写入
    newData = num2cell(t_pred_2d');
end

% 将拼接后的数据写入 Excel 文件
xlswrite(filename, newData);

mae_rmse_data = {mae, rmse, string(elapsedTimeInSeconds)};  % 创建包含 MAE 和 RMSE 以及时间的列
% 写入到 Excel 文件中的另一个 sheet
sheetName = 'Metrics';  % 新的 sheet 名称

if isfile(filename)
    [~, sheets] = xlsfinfo(filename);  % 获取所有 sheet 的名称
    if any(strcmp(sheets, sheetName))
        % 如果 'Metrics' sheet 存在，读取现有数据
        [~, ~, existingMetricsData] = xlsread(filename, sheetName);
        
        % 找到现有数据的最后一行
        nextRow = size(existingMetricsData, 1) + 1;
        
        % 将 MAE 和 RMSE 以及时间追加到现有数据的下一行
        existingMetricsData{nextRow, 1} = mae;
        existingMetricsData{nextRow, 2} = rmse;
        existingMetricsData{nextRow, 3} = string(elapsedTimeInSeconds);
        
        % 将更新后的数据写回 Excel 文件
        xlswrite(filename, existingMetricsData, sheetName);
    else
        % 如果 'Metrics' sheet 不存在，创建它并写入 MAE 和 RMSE 以及时间
        mae_rmse_data_header = {'MAE', 'RMSE', 'Time'};  % 表头
        xlswrite(filename, mae_rmse_data_header, sheetName, 'A1');
        xlswrite(filename, mae_rmse_data, sheetName, 'A2');
    end
else
    % 如果文件不存在，创建新的文件并写入 MAE 和 RMSE 以及时间
    mae_rmse_data_header = {'MAE', 'RMSE', 'Time'};  % 表头
    xlswrite(filename, mae_rmse_data_header, sheetName, 'A1');
    xlswrite(filename, mae_rmse_data, sheetName, 'A2');
end


% 把网络单独保存成 .mat
save('netLSTM.mat','netLSTM','-v7');
%  用 dir 读文件大小
S = dir('netLSTM.mat');
modelSizeMB = S.bytes/1024^2;
fprintf('Model size on disk: %.2f MB\n',modelSizeMB);


end


