clc, clear
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
    % 进行 VMD 分解
    alpha = 2000;   % 惩罚项参数，控制模态分解的平滑程度
    tau = 0;        % 时间步长，设为 0 不进行模式间耦合
    K = 96;          % 分解的模态数（即 VMD 分解的模式数）
    DC = 0;         
    init = 0;       % 模式初始化方法
    tol = 1e-6;     % 误差容忍度

    [u, u_hat, omega] = vmd(variable, alpha, tau, K, DC, init, tol);
    u=u';
    periodicity(:, i) = sum(u(:, 1:5), 2);  
    volatility(:, i) = sum(u(:, 6:end), 2);  
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
