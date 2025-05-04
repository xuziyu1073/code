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


%% 4. 需要手动依次选择列2-7,为了方便进行循环预测。
for i=2:7
variables = data{:,i};
variables = fillmissing(variables, 'constant', 0);

%% 多元序列 单输入
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


%% -- 将数据转化为元胞数组形式 --
% p_train 和 t_train 转换为 1 个元胞，每个元胞包含一个 168xzushu 的矩阵
p_train_cell = mat2cell(p_train, 1, repmat(168, 1, zushu));  % 1 元胞，每个元胞大小为 168xzushu
t_train_cell = mat2cell(t_train, 1, repmat(168, 1, zushu));  % 1 元胞，每个元胞大小为 168xzushu

% p_test 和 t_test 转换为 1 个元胞，每个元胞包含一个 168x1 的矩阵
p_test_cell = mat2cell(p_test, 1, 168);  % 1 元胞，每个元胞大小为 168x1
t_test_cell = t_test;  % 大小为 168x1

%% BP 神经网络构建
% 创建一个前馈神经网络
net = fitnet(72);  % 隐藏层大小可以根据需要调整

% 设置训练参数
net.trainParam.epochs = 500;     % 迭代次数
net.trainParam.lr = 0.001;         % 学习率
% net.performParam.regularization = 0.1;
net.trainParam.showCommandLine = false;  % 禁用命令行输出

%% 训练 BP 神经网络
startTime = datetime('now');
[net, tr] = train(net, p_train_cell, t_train_cell);
endTime = datetime('now');
elapsedTime = endTime - startTime; 

%% 预测
t_pred_cell = net(p_test_cell);  % 针对每个元胞分别进行预测

% 反归一化预测结果
t_pred_cell = cell2mat(t_pred_cell) .* sigma_output + mu_output;  % 反归一化



%% 创建一个新的图形窗口
figure;
% 绘制真实值与预测值
plot(T_test, 'b-', 'LineWidth', 1.5); hold on;  % 绘制真实值（蓝色线）
plot(t_pred_cell, 'r-', 'LineWidth', 1.5);    % 绘制预测值（红色线）
% 添加图例、坐标轴标签和标题
legend('真实值', '预测值', 'Location', 'best');  % 添加图例，标明线条含义
xlabel('样本索引');  % x轴标签
ylabel('数值');  % y轴标签
title('测试集真实值与预测值对比');  % 图表标题
grid on;  % 显示网格
% 保存图形为JPG文件
saveas(gcf, 'comparison_plot.jpg', 'jpg');  % 将图形保存为JPG格式
% 以300 DPI分辨率保存图形
print(gcf, 'comparison_plot', '-djpeg', '-r300');  % 使用gcf获取当前图形，保存为300 DPI分辨率的JPG文件

mae = mean(abs(T_test - t_pred_cell));   % 计算 MAE
rmse = sqrt(mean((T_test - t_pred_cell).^2));  % 计算 RMSE
total_mae = mean(mae);
total_rmse = mean(rmse);

%%
filename = 'T_test_values_BP_单变量.xlsx';
elapsedTimeInSeconds = seconds(elapsedTime);
% 如果文件已经存在，则读取原有数据
if isfile(filename)
    % 读取现有数据
    [~, ~, existingData] = xlsread(filename);
    
    % 将新的 t_pred_cell 数据拼接到现有数据的右边
    newData = [existingData, num2cell(t_pred_cell)];  
else
    % 如果文件不存在，则创建新文件并将 t_pred_cell 写入
    newData = num2cell(t_pred_cell);
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
save('netBP.mat','net','-v7');  
% 用 dir 读文件大小
S = dir('netBP.mat');
modelSizeMB = S.bytes/1024^2;
fprintf('Model size on disk: %.2f MB\n',modelSizeMB);
end