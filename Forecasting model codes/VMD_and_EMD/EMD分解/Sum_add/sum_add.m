clc;
clear;

%% 1. 读取预测数据并进行逐点相加
PeriodicData = xlsread('周期性预测数据_Periodic_forecasting_components.xlsx');
VolatileData = xlsread('波动性预测数据_Volatile_forecasting_components.xlsx');

% 将两个预测数据逐点相加
Proposed_method_data = PeriodicData + VolatileData;  % (168×6)

%% 2. 读取测试数据
Test_data = csvread('EVCSs_test.csv');  % (168×6)

% 分别为 168×6 的矩阵
% % 创建一个文件夹，用于保存图像
outputFolder = 'plots';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

for i = 1:6
    % 每次循环创建一个新窗口
    figure;
    plot(Proposed_method_data(:, i), 'b', 'LineWidth', 1.5); % Proposed_method_data 用蓝色绘制
    hold on;
    plot(Test_data(:, i), 'r', 'LineWidth', 1.5);             % Test_data 用红色绘制
    xlabel('时间索引');
    ylabel('数值');
    title(['Series ' num2str(i) ' 比较']);
    legend({'Proposed Method', 'Test Data'}, 'Location', 'best');
    hold off;
    
    % 保存当前图像到指定文件夹中
    saveas(gcf, fullfile(outputFolder, ['Series_' num2str(i) '.png']));
end

%% 3. 逐列计算 MAE 和 RMSE
mae_row_temp = mean(abs(Proposed_method_data - Test_data), 1);        
rmse_row_temp = sqrt(mean((Proposed_method_data - Test_data).^2,1));
mae_avg = mean(mae_row_temp);
rmse_avg = mean(rmse_row_temp);

%%

% 合并为一个矩阵
output_data = [mae_row_temp'; rmse_row_temp'; mae_avg'; rmse_avg'];

% 将结果保存到 Excel 文件
outputFilename = 'Comparison_results.xlsx';

% 使用 writematrix 函数将数据写入 Excel 文件
writematrix(output_data, outputFilename);


disp(mae_avg)

disp(rmse_avg)


