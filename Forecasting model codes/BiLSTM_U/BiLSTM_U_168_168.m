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

%% ѡ����һ�У����ǵ�����Ԥ��ģ�ͣ���Ҫ�ֶ�����ѡ����2-7,Ϊ�˷������ѭ��Ԥ�⡣
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

%% -- 2) ������ reshape �� 3D --
% ���ݵ���״Ӧ���� [1, 168, zushu]������ 1 �Ǳ�������168 ��ʱ�䲽����zushu ��������
p_train_3d = reshape(p_train, [1, 168, zushu]);  % [1,168,zushu]
t_train_3d = reshape(t_train, [1, 168, zushu]);
p_test_3d  = reshape(p_test,  [1, 168, 1]);
t_test_3d  = reshape(t_test,  [1, 168, 1]);


%% -- ������ reshape �� 3D --
% �������ع�ΪԪ�����飬ÿ��Ԫ������һ��ѵ������
% ÿ�������ĸ�ʽ�� [ʱ�䲽��, ������]
p_train_cell = cell(zushu, 1);  % zushu������
t_train_cell = cell(zushu, 1);  % ��Ӧ��Ŀ������
for i = 1:zushu
    p_train_cell{i} = squeeze(p_train_3d(:, :, i));  % ÿ��ѵ������ [168, 1]
    t_train_cell{i} = squeeze(t_train_3d(:, :, i));  % ÿ��Ŀ������ [168, 1]
end

% ͬ���ع����Լ�����ΪԪ������
p_test_cell = cell(1, 1);
t_test_cell = cell(1, 1);
p_test_cell{1} = squeeze(p_test_3d(:, :, 1));  % �������� [168, 1]
t_test_cell{1} = squeeze(t_test_3d(:, :, 1));  % ����Ŀ������ [168, 1]

%% LSTM ���繹��
inputSize = 1;  % 1*168
numHiddenUnits1 = 72;  % ���ز�ĵ�Ԫ��
outputSize = 1;  % 1*168

layers = [
    sequenceInputLayer(inputSize)  % ����㣬ÿ��ʱ�䲽���������� 1
    bilstmLayer(numHiddenUnits1, 'OutputMode', 'sequence')  % LSTM �㣬����ÿ��ʱ�䲽�����
    dropoutLayer(0.1)  
    fullyConnectedLayer(outputSize)  % ȫ���Ӳ㣬����������� 1
    regressionLayer  % �ع�㣬���ڻع�����
];

% ѵ��ѡ��
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...  
    'MaxEpochs', 500, ...
    'Verbose', 0);

%% ѵ�� LSTM ����
% ʹ��Ԫ��������Ϊ����
startTime = datetime('now');
netLSTM = trainNetwork(p_train_cell, t_train_cell, layers, options);
endTime = datetime('now');
elapsedTime = endTime - startTime; 
elapsedTimeInSeconds = seconds(elapsedTime);
%% Ԥ��
% ʹ�ò������ݽ���Ԥ��
t_pred_cell = predict(netLSTM, p_test_cell);

% ��Ԥ��������һ��
t_pred_2d = t_pred_cell{1} .* sigma_output + mu_output;

%%
mae = mean(abs(T_test - t_pred_2d));   % ���� MAE
rmse = sqrt(mean((T_test - t_pred_2d).^2));  % ���� RMSE
total_mae = mean(mae);
total_rmse = mean(mae);

disp(['Mean Absolute Error (MAE): ', num2str(mae)]);
disp(['Root Mean Squared Error (RMSE): ', num2str(rmse)]);

figure;
plot(T_test, 'b-', 'LineWidth', 1.5); hold on;
plot(t_pred_2d, 'r-', 'LineWidth', 1.5);
legend('��ʵֵ', 'Ԥ��ֵ', 'Location', 'best');
xlabel('��������');
ylabel('��ֵ');
title('���Լ���ʵֵ��Ԥ��ֵ�Ա�');
grid on;

% ����Ϊ300 DPI��JPGͼƬ
saveas(gcf, 'comparison_plot_LSTM.jpg', 'jpg');   % ����ΪJPG��ʽ
print('comparison_plot_LSTM', '-djpeg', '-r300');  % ��300 DPI����

filename = 'T_test_values_BiLSTM_������.xlsx';

% ����ļ��Ѿ����ڣ����ȡԭ������
if isfile(filename)
    % ��ȡ��������
    [~, ~, existingData] = xlsread(filename);
    
    % ���µ� t_pred_2d ����ƴ�ӵ��������ݵ��ұ�
    newData = [existingData, num2cell(t_pred_2d')];  
else
    % ����ļ������ڣ��򴴽����ļ����� t_pred_2d д��
    newData = num2cell(t_pred_2d');
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
save('netLSTM.mat','netLSTM','-v7');
%  �� dir ���ļ���С
S = dir('netLSTM.mat');
modelSizeMB = S.bytes/1024^2;
fprintf('Model size on disk: %.2f MB\n',modelSizeMB);


end


