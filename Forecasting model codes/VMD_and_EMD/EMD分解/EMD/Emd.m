clear
% 加载数据
data = readtable('EVCSs.csv');  % 读取数据

% 提取日期和变量
dates = data.date; 
dates = dates(1:end-168);

variables = data{1:end-168, 2:end}; % 排除日期列，提取变量数据

% 变量的数量
numVariables = size(variables, 2);

% 准备存储周期性和波动性成分
periodicity = zeros(size(variables));
volatility = zeros(size(variables));

for i = 1:numVariables
    % 提取当前变量
    variable = variables(:, i);
    % 进行 EMD 分解
    [imf, residual] = emd(variable);  % EMD 分解
    periodicity(:, i) = sum(imf(:, end-5:end), 2);  % 最后5列的 IMF 是周期性成分，与MSSA重构数量一致
    volatility(:, i) = sum(imf(:, 1:end-6), 2) + residual;  % 其它 IMF 和残差构成波动性成分
end

% 将日期列添加到结果矩阵
periodicity_with_date = [dates, periodicity];
volatility_with_date = [dates, volatility];

% 将结果转换为表格
periodicity_table = array2table(periodicity_with_date, 'VariableNames', ['date', strcat({'EVCS1', 'EVCS2', 'EVCS3', 'EVCS4', 'EVCS5', 'EVCS6'})]);
volatility_table = array2table(volatility_with_date, 'VariableNames', ['date', strcat({'EVCS1', 'EVCS2', 'EVCS3', 'EVCS4', 'EVCS5', 'EVCS6'})]);

% 保存为 Excel 文件
writetable(periodicity_table, 'EVCSs_periodic_components.csv');
writetable(volatility_table, 'EVCSs_volatile_components.csv');
