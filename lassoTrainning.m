%% ============================================================
%  EEG情绪识别：LOSO-CV + L1正则化逻辑回归
%  数据：nTrials × 91
%  第1列=被试ID, 第2-85列=PSD(dB), 第86-90列=波幅, 第91列=标签
% ============================================================
clear; clc; close all;

%% ---------- 1. 加载数据 ----------
% 假设你的数据变量名为 dataAll (nTrials × 91)
% 如果从文件读取，取消下面注释并修改文件名
% dataAll = readmatrix('your_data.csv');
% 或 load('your_data.mat');

% 检查是否有缺失值
load('40dataset.mat');
if any(isnan(Alldata(:)))
    warning('存在NaN，将删除含NaN的试次');
    Alldata(any(isnan(Alldata), 2), :) = [];
end


%% ---------- 2. 拆分数据 ----------
subjectID = Alldata(:, 1);
X         = Alldata(:, 2:90);      % 84列 PSD (已dB)+ 5列 波幅
y         = Alldata(:, 91);        % 标签

nPSD = size(X_psd, 2);
nAmp = size(X_amp, 2);
nFeatures = nPSD + nAmp;




%% ---------- 4. 被试内 z-score 标准化 ----------
uniqueSubjects = unique(subjectID);
nSubjects = length(uniqueSubjects);
X_norm = zeros(size(X));

for s = 1:nSubjects
    idx = (subjectID == uniqueSubjects(s));
    if sum(idx) >= 5   % 至少5个试次才做z-score
        X_norm(idx, :) = zscore(X(idx, :));
    end
end

% 检查标准化结果
fprintf('\n标准化后特征范围: [%.3f, %.3f]\n', min(X_norm(:)), max(X_norm(:)));
fprintf('标准化后每列均值最大绝对值: %.2e\n', max(abs(mean(X_norm))));

%% ---------- 5. 特征名称（用于解释结果）----------
% 请根据你的实际通道名和频段名修改
frontal=[36 37 3 30 17 31 4];%额叶电极
parietal=[22 59	23 7 19 8];%顶叶，文献还多5 32 18 33	6 40 41 ？
occipital=[60 9 61 10];%枕叶
Temporal=[13 14 54 55];%颞叶

COI=[frontal parietal occipital Temporal 5 32 18 33 6 40 41];
COI=chID(COI);
LPPch=chID([7 8 19 59 60]);
bandNames = {'theta','alpha','beta'};

featureNames = cell(1, nFeatures);
for c = 1:length(COI)
    for b = 1:3
        featureNames{(c-1)*3 + b} = sprintf('%s_%s', COI{c}, bandNames{b});
    end
end
for a = 1:nAmp
    featureNames{nPSD + a} = sprintf('%s_Lpp', LPPch{a});
end


%% ---------- 6. LOSO-CV 主循环 ----------
accuracy     = zeros(nSubjects, 1);
y_true_all   = [];
y_pred_all   = [];
y_score_all  = [];
selectedFeatures_all = cell(nSubjects, 1);
bestLambda_all = zeros(nSubjects, 1);
nSelected_all  = zeros(nSubjects, 1);

fprintf('\n========== 开始 LOSO-CV ==========\n');

for s = 1:nSubjects
    testIdx  = (subjectID == uniqueSubjects(s));
    trainIdx = ~testIdx;
    
    X_train = X_norm(trainIdx, :);
    y_train = y(trainIdx);
    X_test  = X_norm(testIdx, :);
    y_test  = y(testIdx);
    
    % ---- 训练 L1 正则化逻辑回归 ----
    % 注意：y_train 必须是 0/1 的列向量
    [B, FitInfo] = lassoglm(X_train, y_train, 'binomial', ...
        'Alpha', 1, ...           % 纯Lasso
        'CV', 5, ...              % 内部5折CV选Lambda
        'Standardize', false);    % 已手动标准化
    
    % 选择偏差最小的Lambda
    [~, bestIdx] = min(FitInfo.Deviance);
    bestLambda = FitInfo.Lambda(bestIdx);
    coef       = B(:, bestIdx);
    intercept  = FitInfo.Intercept(bestIdx);
    
    bestLambda_all(s) = bestLambda;
    selectedFeatures_all{s} = find(coef ~= 0);
    nSelected_all(s) = sum(coef ~= 0);
    
    % ---- 测试集预测 ----
    score  = X_test * coef + intercept;
    y_pred = single(score > 0);
%     prob = 1./(1+exp(-score));    %z > 0 等价于 p > 0.5;预测为正类

    accuracy(s) = mean(y_pred == y_test);
    
    fprintf('被试 %2d/%d | Lambda=%.4f | 选中特征=%2d/%d | 准确率=%.2f%%\n', ...
        s, nSubjects, bestLambda, nSelected_all(s), nFeatures, accuracy(s)*100);
    
    % 累积
    y_true_all  = [y_true_all;  y_test];
    y_pred_all  = [y_pred_all;  y_pred];
    y_score_all = [y_score_all; score];
end

%% ---------- 7. 结果汇总 ----------
fprintf('\n========== LOSO-CV 最终结果 ==========\n');
fprintf('平均准确率:  %.2f%% (± %.2f%%)\n', mean(accuracy)*100, std(accuracy)*100);
fprintf('最高准确率:  %.2f%%\n', max(accuracy)*100);
fprintf('最低准确率:  %.2f%%\n', min(accuracy)*100);
fprintf('总体准确率:  %.2f%%\n', mean(y_pred_all == y_true_all)*100);
fprintf('平均选中特征数: %.1f / %d\n', mean(nSelected_all), nFeatures);

% 二项检验（是否显著高于随机水平50%）
nTotal = length(y_true_all);
nCorrect = sum(y_pred_all == y_true_all);
p_value = 1 - binocdf(nCorrect-1, nTotal, 0.5);
fprintf('二项检验 p = %.4f (vs 随机水平50%%)\n', p_value);

%% ---------- 8. AUC 与 ROC ----------
[Xroc, Yroc, Troc, AUC] = perfcurve(y_true_all, y_score_all, 1);
fprintf('AUC = %.4f\n', AUC);

%% ---------- 9. 可视化 ----------
figure('Position', [50 50 1400 400]);

% 9.1 各被试准确率
subplot(1,3,1);
plot(accuracy*100);
hold on;
yline(mean(accuracy)*100, 'r--', 'LineWidth', 1);
yline(50, 'k:', 'Chance', 'LineWidth', 1);
xlabel('被试'); ylabel('准确率 (%)');
title('各被试 LOSO 准确率');
ylim([0 100]); grid on;

% 9.2 ROC曲线
subplot(1,3,2);
plot(Xroc, Yroc, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--');
xlabel('假阳性率 (FPR)'); ylabel('真阳性率 (TPR)');
title(sprintf('ROC曲线 (AUC=%.3f)', AUC));
grid on;

% 9.3 混淆矩阵
subplot(1,3,3);
cm = confusionmat(y_true_all, y_pred_all);
imagesc(cm);
text(1, 1, num2str(cm(1,1)), 'HorizontalAlignment','center','FontSize',14);
text(2, 1, num2str(cm(2,1)), 'HorizontalAlignment','center','FontSize',14);
text(1, 2, num2str(cm(1,2)), 'HorizontalAlignment','center','FontSize',14);
text(2, 2, num2str(cm(2,2)), 'HorizontalAlignment','center','FontSize',14);
set(gca, 'XTick', [1 2], 'XTickLabel', {'中性(0)','负性(1)'},'YTick', [1 2], 'YTickLabel', {'中性(0)','负性(1)'});
xlabel('预测'); ylabel('真实');
title('混淆矩阵');

sgtitle('LOSO-CV 模型评估');

%% ---------- 10. 特征稳定性分析 ----------
selectionCount = zeros(nFeatures, 1);
for s = 1:nSubjects
    selectionCount(selectedFeatures_all{s}) = ...
        selectionCount(selectedFeatures_all{s}) + 1;
end

[sortedCount, sortedIdx] = sort(selectionCount, 'descend');

fprintf('\n========== 跨被试最稳定特征 (Top 15) ==========\n');
fprintf('%-25s %s\n', '特征名', '被选中次数');
for i = 1:min(15, nFeatures)
    if sortedCount(i) == 0, break; end
    fprintf('%-25s %d/%d\n', featureNames{sortedIdx(i)},sortedCount(i), nSubjects);
end

% 可视化 Top 20
figure('Position', [100 100 900 400]);
topN = min(40, sum(sortedCount > 0));
barh(sortedCount(1:topN));
set(gca, 'YTick', 1:topN,'YTickLabel', featureNames(sortedIdx(1:topN)));
xlabel('被选中次数'); 
title(sprintf('跨被试特征选择稳定性 (Top %d)', topN));
grid on;
set(gca, 'YDir', 'reverse');

%% ---------- 11. 保存结果 ----------
results.accuracy          = accuracy;
results.mean_accuracy     = mean(accuracy);
results.std_accuracy      = std(accuracy);
results.AUC               = AUC;
results.p_value           = p_value;
results.selectedFeatures  = selectedFeatures_all;
results.selectionCount    = selectionCount;
results.featureNames      = featureNames;
results.y_true            = y_true_all;
results.y_pred            = y_pred_all;
results.y_score           = y_score_all;
results.bestLambda        = bestLambda_all;

save('LOSO_results.mat', 'results');
fprintf('\n结果已保存到 LOSO_results.mat\n');