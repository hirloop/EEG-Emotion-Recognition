frontal=[36 37 3 30 17 31 4];%额叶电极
parietal=[22 59	23 7 19 8];%顶叶，文献还多5 32 18 33	6 40 41 ？
occipital=[60 9 61 10];%枕叶
Temporal=[13 14 54 55];%颞叶

COI=[frontal parietal occipital Temporal 5 32 18 33 6 40 41];%全28！（未双侧乳突参考，61导情况）

LppCOI=[7 8 19 59 60];%61导情况


% 载入前测平均参考的图片观看数据（已ICA）;S 1,4,7,17分别为 抑负，观负，抑中，观中。
% 提取每试次3个频段逐个通道的特征值（PSD，Mlpp），并配上被试ID，试次ID，图片类别（标签）
ALLEEG=pop_loadset;AA=pop_loadset;ALLEEG=[ALLEEG AA];

Alldata=[];
for sub=1:length(ALLEEG)
    EEG=ALLEEG(sub);
    %% 提取Mlpp特征值
    %找到epoch分类
    neg=[];
    for eve=1:length(EEG.event)
        if strcmp(EEG.event(eve).type,'S  1') || strcmp(EEG.event(eve).type,'S  4')
            neg=[neg,EEG.event(eve).epoch];
        end
    end
    negEpoch=zeros(1,length(EEG.event));negEpoch(neg)=1;%1为负性，0为中性
    %提取相关通道Mlpp值做特征
    TOI=find(EEG.times>400&EEG.times<600);LppCOI=[7 8 19 59 60];%61导情况
    last5=[squeeze(mean(EEG.data(LppCOI,TOI,:),2))', negEpoch'];  
    
    %% 提取PSD特征
    fs = EEG.srate;
    [~, nTimepoints, nTrials] = size(EEG.data);
    COI=[frontal parietal occipital Temporal 5 32 18 33 6 40 41];%全28！（未双侧乳突参考，61导情况）
    nChannels=COI;
    % 频段
    theta_band = [4 8];
    alpha_band = [8 13];
    beta_band  = [13 30];

    window   = min(256, nTimepoints);
    noverlap = round(window/2);
    nfft     = max(256, 2^nextpow2(window));
    % 每个试次的特征数 = 通道数 × 频段数 (3)
    nFeatures = length(nChannels) * 3;
    X = zeros(nTrials, nFeatures);

    % 逐试次、逐通道计算 PSD
    for t = 1:nTrials
        for c = nChannels
            % 取出该试次该通道的信号 (1 × nTimepoints)
            sig = squeeze(EEG.data(c, :, t));
            
            % 计算功率谱密度
            [pxx, f] = pwelch(sig, window, noverlap, nfft, fs);
            
            % 提取三个频段的平均功率
            theta_idx = (f >= theta_band(1)) & (f < theta_band(2));
            alpha_idx = (f >= alpha_band(1)) & (f < alpha_band(2));
            beta_idx  = (f >= beta_band(1))  & (f < beta_band(2));

            theta_power = mean(pxx(theta_idx));
            alpha_power = mean(pxx(alpha_idx));
            beta_power  = mean(pxx(beta_idx));
            
            % 存入特征矩阵
            % 列顺序: [ch1_theta, ch1_alpha, ch1_beta, ch2_theta, ...]
            colIdx = (find(nChannels==c)-1)*3 + 1;
            X(t, colIdx)   = theta_power;
            X(t, colIdx+1) = alpha_power;
            X(t, colIdx+2) = beta_power;
        end
    end
    X = 10 * log10(X);
    %被试级试次数据合并：subID，28*3 PSD特征，5Mlpp特征，图片类别（01）
    subData=[sub*ones(nTrials,1),X,last5];
    %所有被试数据汇总
    Alldata=[Alldata;subData];
end
