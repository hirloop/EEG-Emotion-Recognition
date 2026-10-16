% 载入数据
load('40dataset.mat');
subjectID = Alldata(:, 1);
X_pre     = Alldata(:, 2:90);      % 89列 PSD (已dB)+波幅
Y         = Alldata(:, 91);        % 标签

% 被试内 z-score 标准化
uniqueSubjects = unique(subjectID);
nSubjects = length(uniqueSubjects);
X_norm = zeros(size(X_pre));

for s = 1:nSubjects
    idx = (subjectID == uniqueSubjects(s));
        X_norm(idx, :) = zscore(X_pre(idx, :));
end

%取lasso所选前n个特征
load('LOSO_results.mat');featureNames=results.featureNames;selectionCount=results.selectionCount;
[sortedCount, sortedIdx] = sort(selectionCount, 'descend');
X=X_norm(:,sortedIdx());

% 1. 划分训练集和测试集 (保持类别比例)
rng(42); % 固定随机种子，保证结果可重复
cv = cvpartition(Y, 'HoldOut', 0.2, 'Stratify', true);
idxTrain = training(cv);
idxTest = test(cv);

X_train = X(idxTrain, :);
Y_train = Y(idxTrain);
X_test = X(idxTest, :);
Y_test = Y(idxTest);

% 2. 训练 SVM 模型
% 这里以内核为 RBF，并开启标准化为例
SVMModel = fitcsvm(X_train, Y_train, ...
    'KernelFunction', 'rbf', ...      % 使用 RBF 核
    'BoxConstraint', 1, ...           % 对应 sklearn 的 C 参数
    'KernelScale', 'auto', ...        % 对应 sklearn 的 gamma，'auto' 为自动估计
    'Standardize', true, ...          % 自动标准化
    'ClassNames', [0; 1]);            % 明确类别标签顺序

% 3. 在测试集上进行预测
[YPred, Score] = predict(SVMModel, X_test);

% 4. 模型评估
% 混淆矩阵
cm = confusionmat(Y_test, YPred);
disp('混淆矩阵:');
disp(cm);

% 计算准确率
accuracy = sum(YPred == Y_test) / length(Y_test);
fprintf('测试集准确率: %.4f\n', accuracy);

% 可视化混淆矩阵
figure;
confusionchart(Y_test, YPred);
title('SVM 混淆矩阵');