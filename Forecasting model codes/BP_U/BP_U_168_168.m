clc;
clear all;
close all
warning off
addpath pathA

%% ��������
rng(0);
data = readtable('EVCSs.csv');
time_str = string(data{:,1});
time = datetime(time_str, 'InputFormat', 'yyyyMMdd');


%% 4. ��Ҫ�ֶ�����ѡ����2-7,Ϊ�˷������ѭ��Ԥ�⡣
for i=2:7
variables = data{:,i};
variables = fillmissing(variables, 'constant', 0);

%% ��Ԫ���� ������
seq = 7; % Ԥ�ⳤ��
split_date = datetime('2023-04-08') + days(7 - seq);
start_date = time(1);
end_date = split_date + days(seq);

P_train = variables(time >= start_date & time < split_date - days(seq), 1:end)';
P_test  = variables(time >= split_date - days(seq) & time < end_date - days(seq), 1:end)';

T_train = variables(time >= start_date + days(seq) & time < split_date, 1:end)';
T_test  = variables(time >= split_date & time < end_date, 1:end)';

%% -- ����������ʹ������ 168 --
len = size(P_train,2);
zushu = floor(len / 168);

p_train = P_train(:, 1:zushu*168);  
t_train = T_train(:, 1:zushu*168);
p_test  = P_test(:, 1:168);
t_test  = T_test(:, 1:168);

%% ���ݹ�һ��
mu_input = mean(p_train, 2);
sigma_input = std(p_train, 0, 2);
p_train = (p_train - mu_input) ./ sigma_input;
p_test  = (p_test  - mu_input) ./ sigma_input;

mu_output = mean(t_train, 2);
sigma_output = std(t_train, 0, 2);
t_train = (t_train - mu_output) ./ sigma_output;
t_test  = (t_test  - mu_output) ./ sigma_output;


%% -- ������ת��ΪԪ��������ʽ --
% p_train �� t_train ת��Ϊ 1 ��Ԫ����ÿ��Ԫ������һ�� 168xzushu �ľ���
p_train_cell = mat2cell(p_train, 1, repmat(168, 1, zushu));  % 1 Ԫ����ÿ��Ԫ����СΪ 168xzushu
t_train_cell = mat2cell(t_train, 1, repmat(168, 1, zushu));  % 1 Ԫ����ÿ��Ԫ����СΪ 168xzushu

% p_test �� t_test ת��Ϊ 1 ��Ԫ����ÿ��Ԫ������һ�� 168x1 �ľ���
p_test_cell = mat2cell(p_test, 1, 168);  % 1 Ԫ����ÿ��Ԫ����СΪ 168x1
t_test_cell = t_test;  % ��СΪ 168x1

%% BP �����繹��
% ����һ��ǰ��������
net = fitnet(72);  % ���ز��С���Ը�����Ҫ����

% ����ѵ������
net.trainParam.epochs = 500;     % ��������
net.trainParam.lr = 0.001;         % ѧϰ��
% net.performParam.regularization = 0.1;
net.trainParam.showCommandLine = false;  % �������������

%% ѵ�� BP ������
startTime = datetime('now');
[net, tr] = train(net, p_train_cell, t_train_cell);
endTime = datetime('now');
elapsedTime = endTime - startTime; 

%% Ԥ��
t_pred_cell = net(p_test_cell);  % ���ÿ��Ԫ���ֱ����Ԥ��

% ����һ��Ԥ����
t_pred_cell = cell2mat(t_pred_cell) .* sigma_output + mu_output;  % ����һ��



%% ����һ���µ�ͼ�δ���
figure;
% ������ʵֵ��Ԥ��ֵ
plot(T_test, 'b-', 'LineWidth', 1.5); hold on;  % ������ʵֵ����ɫ�ߣ�
plot(t_pred_cell, 'r-', 'LineWidth', 1.5);    % ����Ԥ��ֵ����ɫ�ߣ�
% ���ͼ�����������ǩ�ͱ���
legend('��ʵֵ', 'Ԥ��ֵ', 'Location', 'best');  % ���ͼ����������������
xlabel('��������');  % x���ǩ
ylabel('��ֵ');  % y���ǩ
title('���Լ���ʵֵ��Ԥ��ֵ�Ա�');  % ͼ�����
grid on;  % ��ʾ����
% ����ͼ��ΪJPG�ļ�
saveas(gcf, 'comparison_plot.jpg', 'jpg');  % ��ͼ�α���ΪJPG��ʽ
% ��300 DPI�ֱ��ʱ���ͼ��
print(gcf, 'comparison_plot', '-djpeg', '-r300');  % ʹ��gcf��ȡ��ǰͼ�Σ�����Ϊ300 DPI�ֱ��ʵ�JPG�ļ�

mae = mean(abs(T_test - t_pred_cell));   % ���� MAE
rmse = sqrt(mean((T_test - t_pred_cell).^2));  % ���� RMSE
total_mae = mean(mae);
total_rmse = mean(rmse);

%%
filename = 'T_test_values_BP_������.xlsx';
elapsedTimeInSeconds = seconds(elapsedTime);
% ����ļ��Ѿ����ڣ����ȡԭ������
if isfile(filename)
    % ��ȡ��������
    [~, ~, existingData] = xlsread(filename);
    
    % ���µ� t_pred_cell ����ƴ�ӵ��������ݵ��ұ�
    newData = [existingData, num2cell(t_pred_cell)];  
else
    % ����ļ������ڣ��򴴽����ļ����� t_pred_cell д��
    newData = num2cell(t_pred_cell);
end

% ��ƴ�Ӻ������д�� Excel �ļ�
xlswrite(filename, newData);

mae_rmse_data = {mae, rmse, string(elapsedTimeInSeconds)};  % �������� MAE �� RMSE �Լ�ʱ�����
% д�뵽 Excel �ļ��е���һ�� sheet
sheetName = 'Metrics';  % �µ� sheet ����

if isfile(filename)
    [~, sheets] = xlsfinfo(filename);  % ��ȡ���� sheet ������
    if any(strcmp(sheets, sheetName))
        % ��� 'Metrics' sheet ���ڣ���ȡ��������
        [~, ~, existingMetricsData] = xlsread(filename, sheetName);
        
        % �ҵ��������ݵ����һ��
        nextRow = size(existingMetricsData, 1) + 1;
        
        % �� MAE �� RMSE �Լ�ʱ��׷�ӵ��������ݵ���һ��
        existingMetricsData{nextRow, 1} = mae;
        existingMetricsData{nextRow, 2} = rmse;
        existingMetricsData{nextRow, 3} = string(elapsedTimeInSeconds);
        
        % �����º������д�� Excel �ļ�
        xlswrite(filename, existingMetricsData, sheetName);
    else
        % ��� 'Metrics' sheet �����ڣ���������д�� MAE �� RMSE �Լ�ʱ��
        mae_rmse_data_header = {'MAE', 'RMSE', 'Time'};  % ��ͷ
        xlswrite(filename, mae_rmse_data_header, sheetName, 'A1');
        xlswrite(filename, mae_rmse_data, sheetName, 'A2');
    end
else
    % ����ļ������ڣ������µ��ļ���д�� MAE �� RMSE �Լ�ʱ��
    mae_rmse_data_header = {'MAE', 'RMSE', 'Time'};  % ��ͷ
    xlswrite(filename, mae_rmse_data_header, sheetName, 'A1');
    xlswrite(filename, mae_rmse_data, sheetName, 'A2');
end
% �����絥������� .mat
save('netBP.mat','net','-v7');  
% �� dir ���ļ���С
S = dir('netBP.mat');
modelSizeMB = S.bytes/1024^2;
fprintf('Model size on disk: %.2f MB\n',modelSizeMB);
end