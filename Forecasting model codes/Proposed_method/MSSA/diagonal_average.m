
%% 分割重构的轨迹矩阵回各变量，并进行对角线平均
function recon_time_series = diagonal_average(Y_recon, num_variables, M, K)
    % diagonal_average - 对轨迹矩阵进行对角线平均，重构时间序列
    %
    % 输入参数：
    %   Y_recon         - 重构后的轨迹矩阵 (M x (K * num_variables))
    %   num_variables   - 变量数量
    %   M               - 窗口长度
    %   K               - 每个变量的轨迹数
    %
    % 输出参数：
    %   recon_time_series - 重构后的时间序列 (N x num_variables)

    N = M + K - 1;
    L_star = min(M, K);
    K_star = max(M, K);
    
    recon_time_series = zeros(N, num_variables);
    
    for m = 1:num_variables
        % 提取每个变量的重构轨迹矩阵
        X_recon_m = Y_recon(:, (m-1) * K + 1 : m * K);  % (M x K)
        
        % 对角线平均
        recon_m = zeros(N, 1);
        
        % 第一阶段：1 <= n_c <= L*
        for nc = 1:L_star
            for s = 1:nc
                recon_m(nc) = recon_m(nc) + X_recon_m(s, nc - s + 1);
            end
            recon_m(nc) = recon_m(nc) / nc;  % 标准化
        end
        
        % 第二阶段：L* <= n_c <= K*
        for nc = L_star + 1:K_star
            for s = 1:L_star
                recon_m(nc) = recon_m(nc) + X_recon_m(s, nc - s + 1);
            end
            recon_m(nc) = recon_m(nc) / L_star;  % 标准化
        end
        
        % 第三阶段：K* <= n_c <= N
        for nc = K_star + 1:N
            for s = nc - K_star + 1:M
                recon_m(nc) = recon_m(nc) + X_recon_m(s, nc - s + 1);
            end
            recon_m(nc) = recon_m(nc) / (N - nc + 1);  % 标准化
        end
        
        recon_time_series(:, m) = recon_m;
    end
end
