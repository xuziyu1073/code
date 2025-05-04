clc;
clear;

%% 1. ��ȡԤ�����ݲ�����������
PeriodicData = xlsread('������Ԥ������_Periodic_forecasting_components.xlsx');
VolatileData = xlsread('������Ԥ������_Volatile_forecasting_components.xlsx');

% ������Ԥ������������
Proposed_method_data = PeriodicData + VolatileData;  % (168��6)

%% 2. ��ȡ��������
Test_data = csvread('EVCSs_test.csv');  % (168��6)

% �ֱ�Ϊ 168��6 �ľ���
% % ����һ���ļ��У����ڱ���ͼ��
outputFolder = 'plots';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

for i = 1:6
    % ÿ��ѭ������һ���´���
    figure;
    plot(Proposed_method_data(:, i), 'b', 'LineWidth', 1.5); % Proposed_method_data ����ɫ����
    hold on;
    plot(Test_data(:, i), 'r', 'LineWidth', 1.5);             % Test_data �ú�ɫ����
    xlabel('ʱ������');
    ylabel('��ֵ');
    title(['Series ' num2str(i) ' �Ƚ�']);
    legend({'Proposed Method', 'Test Data'}, 'Location', 'best');
    hold off;
    
    % ���浱ǰͼ��ָ���ļ�����
    saveas(gcf, fullfile(outputFolder, ['Series_' num2str(i) '.png']));
end

%% 3. ���м��� MAE �� RMSE
mae_row_temp = mean(abs(Proposed_method_data - Test_data), 1);        
rmse_row_temp = sqrt(mean((Proposed_method_data - Test_data).^2,1));
mae_avg = mean(mae_row_temp);
rmse_avg = mean(rmse_row_temp);

%%

% �ϲ�Ϊһ������
output_data = [mae_row_temp'; rmse_row_temp'; mae_avg'; rmse_avg'];

% ��������浽 Excel �ļ�
outputFilename = 'Comparison_results.xlsx';

% ʹ�� writematrix ����������д�� Excel �ļ�
writematrix(output_data, outputFilename);


disp(mae_avg)

disp(rmse_avg)


