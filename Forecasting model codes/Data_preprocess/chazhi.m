clc;
clear;

%% 1. 读取和预处理数据
% 读取CSV文件
filename = 'originaldata.csv';

data1 = readtable(filename);

% 预测的7日负荷数据未知，无法对其进行插值或分解处理
data = data1(1:end-7*24,:);
data1 = data1(end-7*24+1:end,:);
% 获取唯一的日期（假设格式为yyyyMMdd）
uniqueDates = unique(data.date);
numDates = length(uniqueDates);
numHours = 24;

% 生成包含具体时间的datetime向量
% 重复每个唯一日期24次
repeatedDates = repelem(datetime(string(uniqueDates), 'InputFormat', 'yyyyMMdd'), numHours, 1);

% 创建小时向量（0到23，重复每个日期）
hoursVec = repmat((0:numHours-1)', numDates, 1);

% 生成完整的datetime向量
datetimeArray = repeatedDates + hours(hoursVec);

% 检查生成的datetime向量长度是否与数据行数匹配
if length(datetimeArray) ~= height(data)
    error('生成的datetime向量长度与数据行数不匹配。');
end

% 将生成的datetime向量赋值给变量dateTime
dateTime = datetimeArray;

% 提取EVCS列（假设列名为EVXS1到EVCS6）
EVCS = data{:, {'EVCS1', 'EVCS2', 'EVCS3', 'EVCS4', 'EVCS5', 'EVCS6'}};

%% 2. 识别异常日并进行插值
% 定义可调节的日期范围（用户可根据需要修改）
% 例如，绘制2022年4月15日至2023年4月07日的数据
startDate = datetime('2022-04-15', 'Format', 'yyyy-MM-dd');  % 起始日期
endDate = datetime('2023-04-07', 'Format', 'yyyy-MM-dd') + hours(23);    % 结束日期，包含当天24小时

% 过滤数据以仅包含在指定日期范围内的记录
dateFilter = (dateTime >= startDate) & (dateTime <= endDate);
filteredDate = dateTime(dateFilter);
filteredEVCS = EVCS(dateFilter, :);

% 检查是否有数据在指定范围内
if isempty(filteredDate)
    error('在指定的日期范围内没有找到数据。请检查日期范围或数据文件。');
end

% 初始化插值后的EVCS矩阵
EVCS_interpolated = filteredEVCS;

% EVCS列名称和对应的纵坐标标签
EVCS_labels = {'EVCS1', 'EVCS2', 'EVCS3', ...
             'EVCS4', 'EVCS5', 'EVCS6'};

% 创建一个新的图形窗口
figure;

% 创建6行1列的布局
t_RC = tiledlayout(6,1, 'TileSpacing', 'none', 'Padding', 'none');

for EVCSIdx = 1:6
    % 当前EVCS列的数据
    currentEVCS = filteredEVCS(:, EVCSIdx);
    
    % 计算整体平均值
    overallMean = mean(currentEVCS);
    
    % 计算总点数和总天数
    numTotalPoints = length(currentEVCS);
    numDays = numTotalPoints / 24;
    
    % 重塑数据为[numDays, 24]的矩阵，每行对应一天的数据
    dailyData = reshape(currentEVCS, 24, numDays)';
    
    % 计算每日平均值
    dailyMean = mean(dailyData, 2);
    
    %% 修改开始：根据EVCSIdx调整异常日阈值
    % 标识每日平均值低于整体平均值的异常日
    if EVCSIdx == 3
        abnormalDays = dailyMean < overallMean * 0.5;
    else
        abnormalDays = dailyMean < overallMean * 0.3;
    end
    %% 修改结束
    
    % 获取异常日和正常日的索引
    abnormalDateIndices = find(abnormalDays);
    normalDateIndices = find(~abnormalDays);
    
    if isempty(abnormalDateIndices)
        warning(['没有检测到EVCS' num2str(EVCSIdx) '列的任何异常日。']);
    end
    
    % 为每个异常日找到最近的7个正常日并计算每小时平均负荷
    numAbnormalDays = length(abnormalDateIndices);
    interpolatedData = NaN(numAbnormalDays, numHours);
    
    for abIdx = 1:numAbnormalDays
        abDateIdx = abnormalDateIndices(abIdx);
        abDate = uniqueDates(abDateIdx);
        abDateTime = datetime(string(abDate), 'InputFormat', 'yyyyMMdd');
        
        % 计算所有正常日与当前异常日的时间差
        normalDates = uniqueDates(normalDateIndices);
        normalDateTimes = datetime(string(normalDates), 'InputFormat', 'yyyyMMdd');
        timeDiff = abs(days(abDateTime - normalDateTimes));
        
        % 找到时间差最小的7个正常日
        [~, sortedIndices] = sort(timeDiff);
        nearest7Indices = sortedIndices(1:min(7, length(sortedIndices)));
        nearest7DateIndices = normalDateIndices(nearest7Indices);
        
        % 提取这7个正常日的负荷数据
        nearest7Data = dailyData(nearest7DateIndices, :);  % [7, 24]
        
        % 计算每小时的平均负荷
        hourlyAvg = mean(nearest7Data, 1);  % [1, 24]
        
        % 存储插值数据
        interpolatedData(abIdx, :) = hourlyAvg;
    end
    
    % 替换异常日的数据为插值数据
    for abIdx = 1:numAbnormalDays
        abDateIdx = abnormalDateIndices(abIdx);
        dayStart = (abDateIdx - 1) * numHours + 1;
        dayEnd = abDateIdx * numHours;
        EVCS_interpolated(dayStart:dayEnd, EVCSIdx) = interpolatedData(abIdx, :)';
    end
    
    %% 3. 绘图
    % 使用 nexttile 代替 subplot
    ax = nexttile;
    hold on;
    
    % 初始化句柄变量
    hNormal = [];
    hAbnormal = [];
    hInterpolated = [];
    
    % 绘制所有正常日的数据为浅蓝色实线
    for i = 1:numDates
        if ~abnormalDays(i)
            dayStart = (i-1)*numHours + 1;
            dayEnd = i*numHours;
            h = plot(filteredDate(dayStart:dayEnd), currentEVCS(dayStart:dayEnd), 'Color', [0.6 0.8 1], 'LineStyle', '-', 'LineWidth', 0.25);
            if isempty(hNormal)
                hNormal = h;  % 记录第一个正常数据的句柄
            end
        end
    end
    
    % 绘制插值后的数据为浅红色实线
    for abIdx = 1:numAbnormalDays
        abDateIdx = abnormalDateIndices(abIdx);
        dayStart = (abDateIdx-1)*numHours + 1;
        dayEnd = abDateIdx*numHours;
        h = plot(filteredDate(dayStart:dayEnd), EVCS_interpolated(dayStart:dayEnd, EVCSIdx), 'Color', [1 0.5 0.5], 'LineStyle', '-', 'LineWidth', 0.25);
        if isempty(hInterpolated)
            hInterpolated = h;  % 记录第一个插值数据的句柄
        end
    end
    
    hold off;
    xlim([filteredDate(1), filteredDate(end)]);
    % 设置纵坐标标签
    ylabel(EVCS_labels{EVCSIdx}, 'FontSize', 8, 'FontName', 'Times New Roman');
    ylim([0 1000])
    
    % 设置字体为Times New Roman的坐标轴刻度
    ax.FontName = 'Times New Roman';
    
    % 修改x轴标签显示
    if EVCSIdx == 6
        xlabel('Time (h)', 'FontName', 'Times New Roman', 'FontSize', 10);
        
        % 计算5个均匀分布的索引
        n = length(filteredDate);
        indices = round(linspace(1, n, 5));
        
        % 确保索引不超出范围
        indices(indices < 1) = 1;
        indices(indices > n) = n;
        
        % 设置xticks为第一个、25%、50%、75%和最后一个日期
        xticks(ax, filteredDate(indices));
        
        % 设置日期格式
        ax.XAxis.TickLabelFormat = 'dd/MM/yyyy 00:00';
    else
        % 计算5个均匀分布的索引
        n = length(filteredDate);
        indices = round(linspace(1, n, 5));
        
        % 确保索引不超出范围
        indices(indices < 1) = 1;
        indices(indices > n) = n;
        
        % 设置xticks为第一个、25%、50%、75%和最后一个日期
        xticks(ax, filteredDate(indices));
        
        % 隐藏横坐标刻度标签
        xticklabels(ax, []);  % 隐藏横坐标刻度标签
        xlabel(ax, '');  % 隐藏横坐标标签
    end
    
%     grid on;  % 添加网格以提高可读性
    
    % 设置坐标轴为全封闭格式
    set(gca, 'Box', 'on');
    
    % 添加图例
    if EVCSIdx == 1
        % 确保句柄不为空
        if isempty(hNormal)
            hNormal = plot(NaN, NaN, 'Color', [0.6 0.8 1], 'LineStyle', '-', 'LineWidth', 0.25);
        end
        if isempty(hInterpolated)
            hInterpolated = plot(NaN, NaN, 'Color', [1 0.5 0.5], 'LineStyle', '-', 'LineWidth', 0.25);
        end
        
        % 创建图例
        legend([hNormal,  hInterpolated], {'Normal data', 'Interpolated data'}, 'Location', 'best', 'FontName', 'Times New Roman');
    end
end
ylabel(t_RC, 'Charging load (kW)', 'FontSize', 10, 'FontName', 'Times New Roman');

% 优化子图布局以防止标签重叠
% 由于使用了 tiledlayout 并设置了 'TileSpacing' 和 'Padding' 为 'none'，
% 通常不需要进一步调整图形窗口大小。但您可以根据需要调整：
set(gcf, 'Position', [100, 100, 500, 600]);  % 根据需要调整图形窗口大小


%% 4. 保存插值后的数据 训练集
% 将插值后的数据保存到新的CSV文件中
newDatatrain = data;
newDatatrain{dateFilter, {'EVCS1', 'EVCS2', 'EVCS3', 'EVCS4', 'EVCS5', 'EVCS6'}} = EVCS_interpolated;
writetable(newDatatrain, 'EVCSs_train.csv');  % 保存为 EVCSs_train.csv



%% 4. 保存插值后的数据 测试集
newDatatest = data1;
writetable(newDatatest, 'EVCSs_test.csv');  % 保存为 EVCSs_test.csv


%% 4. 保存插值后的数据 原数据
newData = [newDatatrain; newDatatest];
writetable(newData, 'EVCSs.csv');  % 保存为 EVCSs_train.csv


%% 5. 保存图像为PNG格式，分辨率300 dpi
% 确保所有图形渲染完成
drawnow;

% 保存当前图形为PNG，300 dpi
print('EVCS_Loads_Plot-全部.png','-dpng','-r300');
