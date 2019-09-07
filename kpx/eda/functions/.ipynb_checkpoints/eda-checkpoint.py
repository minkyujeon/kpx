import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def mutual_info_table(df) : 
    temp = pd.DataFrame(columns=df.columns)
    for index in range(len(df.columns)) : 
        temp.loc[index] = np.zeros(len(df.columns))
    temp.index = df.columns
    
    for i, column in enumerate(df.columns) : 
        for j, index in enumerate(df.columns) :
            temp.iloc[i, j] = normalized_mutual_info_score(df[df.columns[i]], df[df.columns[j]])
            
    return temp

def show_correlation(df, how_many=5) :
    dataCorr = df.corr()
    dataCorr = dataCorr.mask(np.tril(np.ones(dataCorr.shape)).astype(np.bool))

    print('[Correlation : Descending Order]')
    print(dataCorr.unstack().sort_values(ascending=False)[:how_many])
    print('\n[Correlation : Ascending Order]')
    print(dataCorr.unstack().sort_values()[:how_many])

    plt.figure(figsize=(10,10))
    sns.heatmap(df.corr(), annot=True, fmt='.2f')
    plt.title('Correlation')
    plt.show()
    return

def show_normalized_mutual_information(df, how_many=5) :
    dataMInf_table = mutual_info_table(df)
    dataMInf = dataMInf_table.mask(np.tril(np.ones(dataMInf_table.shape)).astype(np.bool))


    print('[Normalized Mutual Information : Descending Order]')
    print(dataMInf.unstack().sort_values(ascending=False)[:how_many])
    
    plt.figure(figsize=(10,10))
    sns.heatmap(dataMInf_table, annot=True, fmt='.2f')
    plt.title('Normalized Mutual Information')
    plt.show()
    return

def get_bins(df, by, target, num_bins) :
   # by : x (interval)
   # target : y (mean per interval)
    means = np.zeros(num_bins)
    medians = np.zeros(num_bins)
    stds = np.zeros(num_bins)

    interval = (df[by].max() - df[by].min())/num_bins
    print('interval:',interval)
    scales = np.linspace(df[by].min(), df[by].max(), num=num_bins)

    for i in range(num_bins) :
        indices = df[df[by] <interval*(i+1)]
        indices = indices[indices[by] >= interval * i]
        print('indices:',indices)
        medians[i] = indices[target].median()
        means[i] = indices[target].mean()
        stds[i] = indices[target].std()

    sns.scatterplot(scales, means, alpha=0.6, s=30, label='mean')
    sns.scatterplot(scales, medians, alpha=0.6, s=30, label='median')
    sns.lineplot(scales, means+stds, alpha=0.2, label='mean+1std',color='green')
    sns.lineplot(scales, means-stds, alpha=0.2, label='mean-1std',color='green')
    print('means:',means)
    print('scales:',scales)
    plt.xlabel(by + ' (%d Bins)'%num_bins)
    plt.ylabel(target)
    plt.legend()
    plt.title('Mean Y value per X interval')
    plt.show()

    return

def relative_density_plot(x, y, num_bins, label=['x', 'Power Generation(kW)'], show=False) :
    # x : x로 들어가는것, 구간별로 쪼개져야 함
    # y : y로 보고싶은것, 구간별로 mean을 뱉어야 함
    
    # 전체 길이 index list 생성
    scale = x.max() - x.min()
    interval = scale / num_bins
    scales = np.linspace(x.min(), x.max(), num=num_bins)
    
    # means, medians, stds
    means = np.zeros(num_bins)
    medians = np.zeros(num_bins)
    stds = np.zeros(num_bins)
    maxes = np.zeros(num_bins)
    mins = np.zeros(num_bins)
    
    # 인덱스 초기화
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    for idx_bin in range(num_bins) :
        left = x.min() + interval * idx_bin
        right = x.min() + interval * (idx_bin+1)
        # 이 x 구간에 들어있는 indice로
        # y 값들을 뽑아야 함
        t = y[x.between(left, right)]
        if t.shape[0] == 0 :
            means[idx_bin] = np.nan
            medians[idx_bin] = np.nan
            stds[idx_bin] = np.nan
            maxes[idx_bin] = np.nan
            mins[idx_bin] = np.nan
        else :
            means[idx_bin] = t.mean()
            medians[idx_bin] = t.median()
            stds[idx_bin] = t.std()
            maxes[idx_bin] = t.max()
            mins[idx_bin] = t.min()
    
    sns.scatterplot(scales, means, alpha=0.9, s=30, label='mean')
    sns.scatterplot(scales, medians, alpha=0.9, s=30, label='median')
    sns.lineplot(scales, means+stds, alpha=0.1, label='mean+-1std',color='green')
    sns.lineplot(scales, means-stds, alpha=0.1, color='green')
    sns.lineplot(scales, maxes, alpha=0.1, label='max/min', color='red')
    sns.lineplot(scales, mins, alpha=0.1, color='red')
        
    plt.legend()
    plt.title('Mean Y value per X interval')
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    if show == True : 
        plt.show()
    
    return


def max_density_plot(x, y, num_bins, label=['x', 'y'], title ='', show=True) :
   # x : x로 들어가는것, 구간별로 쪼개져야 함
   # y : y로 보고싶은것, 구간별로 mean을 뱉어야 함

   # 전체 길이 index list 생성
    scale = x.max() - x.min()
    interval = scale / num_bins
    scales = np.linspace(x.min(), x.max(), num=num_bins)

    # means, medians, stds
    maxes = np.zeros(num_bins)


    # 인덱스 초기화
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    for idx_bin in range(num_bins) :
        left = x.min() + interval * idx_bin
        right = x.min() + interval * (idx_bin+1)
        # 이 x 구간에 들어있는 indice로
        # y 값들을 뽑아야 함
        t = y[x.between(left, right)]
        if t.shape[0] == 0 :
            maxes[idx_bin] = np.nan
        else :
            maxes[idx_bin] = t.max()

    sns.lineplot(scales, maxes, alpha=0.9, label='max/min', color='red')


    plt.legend()
    plt.title(title)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    if show == True :
        plt.show()

    return scales, maxes



def show_relative_density_plot(df, target) :
    for idx, column in enumerate(df.columns) :
        # set subplots
        if idx%3 == 0:
            plt.figure(figsize=(15, 3))
        plt.subplot(1, 3, (idx%3)+1)

        # plot
        if column == 'Wind Direction(16)' :
            num_bins = 16 #16방위
        elif column == 'Cloud' :
            num_bins = 10
        elif column == 'year' :
            num_bins = 3
        elif column == 'month' :
            num_bins = 12
        elif column == 'day' :
            num_bins = 31
        elif column == 'hour' :
            num_bins = 24
        else : 
            num_bins = 100
        
        by = column
        
        relative_density_plot(df[column],df[target],num_bins)
#         means = np.zeros(num_bins)
#         stds = np.zeros(num_bins)
        
#         interval = (df[by].max() - df[by].min())/num_bins
#         scales = np.linspace(df[by].min(), df[by].max(), num=num_bins)

#         for i in range(num_bins) :
#             indices = df[df[by] < interval*(i+1)]
#             indices = indices[indices[by] >= interval*i]
            
#             means[i] = indices[target].mean()
#             stds[i] = indices[target].std()

#         sns.scatterplot(scales, means, alpha=0.9, s=30, label='mean')
#         #plt.plot(scales, means - stds, label='mean-1std', color='green', alpha=0.2)
#         #plt.plot(scales, means + stds, label='mean+1std', color='green', alpha=0.2)  
#         sns.lineplot(scales, means+stds, alpha=0.2, label='mean+1std',color='green')
#         sns.lineplot(scales, means-stds, alpha=0.2, label='mean-1std',color='green')
        
        plt.xlabel(by + ' (%d Bins)'%num_bins)
        plt.ylabel(target)
        plt.legend()
        plt.title('Mean Y value per X interval')


        # show subplots
        if idx%3 == 2 :
            plt.tight_layout()
            plt.show()
    return
                