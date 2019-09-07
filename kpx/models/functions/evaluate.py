from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def relative_density_plot(x, y, num_bins, label=['x', 'y'], show=False) :
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
    
    sns.scatterplot(scales, means, alpha=0.9, s=30, label='train_mean')
#     sns.scatterplot(scales, medians, alpha=0.9, s=30, label='median')
    sns.lineplot(scales, means+stds, alpha=0.1, label='train_mean+-1std',color='green')
    sns.lineplot(scales, means-stds, alpha=0.1, color='green')
    sns.lineplot(scales, maxes, alpha=0.1, label='train_max/min', color='red')
    sns.lineplot(scales, mins, alpha=0.1, color='red')
        
    plt.legend()
    plt.title('Mean Y value per X interval')
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    if show == True : 
        plt.show()
    
    return

class EnsembledRegressor() :
    def __init__(self, model_list) :
        self.names = [i[0] for i in model_list]
        self.models = [i[1] for i in model_list]
        self.num_models = len(model_list)
        # 혹은 []
        
    def fit(self, x, y) :
        for model in self.models :
            model.fit(x, y)
            
    def predict(self, x) :
        predictions = np.zeros(x.shape[0])
        for model in self.models :
            predictions += model.predict(x)
        predictions /= self.num_models
        # return average(prediction)
        return predictions
    
    def predict_all(self, x) :
        result = []
        for idx in range(self.num_models) :
            result.append((self.names[idx], self.models[idx].predict(x)))
        # return list of tuple
        # [('model name', prediction), ('model name', prediction), ...]
        return result 
    
def nMAE(y,yhat,m=45000):
    return np.abs((yhat-y)/m).mean()

def evaluate_idea_2(df, model_list) :
    # ensembled model
    model = EnsembledRegressor(model_list)
    names = [i[0] for i in model_list]
    num_models = len(model_list)
    
    # result dataframe
    df_result = pd.DataFrame(columns=['model',
                                     'train r2 (time)',
                                     'test r2 (time)',
                                     'train r2 (random)',
                                     'test r2 (random)',
                                     'train r2 std (time)',
                                     'test r2 std (time)',
                                     'train r2 std (random)',
                                     'test r2 std (random)'])
    df_result['model'] = df_result['model'].astype(str)
    df_result = df_result.append({'model' : 'average'}, ignore_index=True)
    for name in names :
        df_result = df_result.append({'model' : name}, ignore_index=True)
    
    # x, y
    df = df.reset_index(drop=True)
    x = df.drop(['datetime', 'date', 'Power Generation(kW)'], axis=1)
    #x = df.drop(['Power Generation(kW)'], axis=1)
    y = df['Power Generation(kW)']
    
    # KFold Validation
    n_splits = 5
    
    #################################################################
    #  Random Split
    kf = KFold(n_splits=n_splits, shuffle=True)
    ## average
    train_r2 = np.zeros(n_splits)
    test_r2 = np.zeros(n_splits)
    ## for each model
    train_r2_models = np.zeros((num_models, n_splits))
    test_r2_models = np.zeros((num_models, n_splits))
    
    
    for idx, (train_index, test_index) in enumerate(kf.split(x)) :

        # split data
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        datetime_train, datetime_test = df['date'].iloc[train_index], df['date'].iloc[test_index]
        # train
        model.fit(x_train, y_train)
        
        # evaluate
        
        ## each model score
        y_train_pred_all = model.predict_all(x_train)
        y_test_pred_all = model.predict_all(x_test)
        train_predictions = [i[1] for i in y_train_pred_all]
        test_predictions = [i[1] for i in y_test_pred_all]
        for idx_model, train_prediction in enumerate(train_predictions) : 
            train_r2_models[idx_model][idx] = r2_score(y_train, train_prediction)
        for idx_model, test_prediction in enumerate(test_predictions) : 
            test_r2_models[idx_model][idx] = r2_score(y_test, test_prediction)

        ## average score
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        train_r2[idx] = r2_score(y_train, y_train_pred)
        test_r2[idx] = r2_score(y_test, y_test_pred)
        #########
        print('random')
        mse_train = (y_train_pred - y_train)**2
        mse_test = (y_test_pred - y_test)**2

        nMAE_train = nMAE(y_train, y_train_pred)
        nMAE_test = nMAE(y_test, y_test_pred)

        mse = pd.concat([mse_train, mse_test]).reset_index(drop=False)
        #display(mse)
        plt.figure(figsize=(20, 6))
        relative_density_plot(mse['index'], mse['Power Generation(kW)'], num_bins=1000, show=False)
        plt.plot(mse['index'], mse['Power Generation(kW)'], color = 'green')
        plt.axvline(x=mse_train.shape[0])
        plt.legend()
        plt.show()

        #######
        power = pd.concat([y_train, y_test]).reset_index(drop=True)
        datetime = pd.concat([datetime_train, datetime_test]).reset_index(drop=True)
        plt.figure(figsize=(20,6))
        plt.plot(datetime, power, label='original',alpha=0.2)
        plt.plot(datetime, power.rolling(48).mean(), label='ma_48',alpha=0.7)
        plt.plot(datetime, power.rolling(150).mean(), label='ma_150',alpha=0.7)
        plt.plot(datetime, power.rolling(600).mean(), label='ma_600',alpha=0.7)
        plt.axvline(x=datetime_train.iloc[-1])
        plt.legend()
        plt.show()      
#         print('train_r2_random :', train_r2[idx])
#         print('test_r2_random :', test_r2[idx])
#         print('train_nMAE_random :',nMAE(y_train, y_train_pred))
#         print('test_nMAE_random :',nMAE(y_test, y_test_pred))
        
    df_result['train r2 (random)'][0] = train_r2.mean()
    df_result['test r2 (random)'][0] = test_r2.mean()
    df_result['train r2 std (random)'][0] = train_r2.std()
    df_result['test r2 std (random)'][0] = test_r2.std()
    
    for idx in range(1, num_models+1) :
        df_result['train r2 (random)'][idx] = train_r2_models[idx-1].mean()
        df_result['test r2 (random)'][idx] = test_r2_models[idx-1].mean()
        df_result['train r2 std (random)'][idx] = train_r2_models[idx-1].std()
        df_result['test r2 std (random)'][idx] = test_r2_models[idx-1].std()
    
    
    #################################################################
    # Time Split
    kf = KFold(n_splits=n_splits, shuffle=False)
    ## average
    train_r2 = np.zeros(n_splits)
    test_r2 = np.zeros(n_splits)
    ## for each model
    train_r2_models = np.zeros((num_models, n_splits))
    test_r2_models = np.zeros((num_models, n_splits))
    
    
    for idx, (train_index, test_index) in enumerate(kf.split(x)) :

        # split data
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        datetime_train, datetime_test = df['date'].iloc[train_index], df['date'].iloc[test_index]
        # train
        model.fit(x_train, y_train)
        
        # evaluate
        
        ## each model score
        y_train_pred_all = model.predict_all(x_train)
        y_test_pred_all = model.predict_all(x_test)
        train_predictions = [i[1] for i in y_train_pred_all]
        test_predictions = [i[1] for i in y_test_pred_all]
        for idx_model, train_prediction in enumerate(train_predictions) : 
            train_r2_models[idx_model][idx] = r2_score(y_train, train_prediction)
        for idx_model, test_prediction in enumerate(test_predictions) : 
            test_r2_models[idx_model][idx] = r2_score(y_test, test_prediction)

        ## average score
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        train_r2[idx] = r2_score(y_train, y_train_pred)
        test_r2[idx] = r2_score(y_test, y_test_pred)
        
        #########
        print('time')
        mse_train = (y_train_pred - y_train)**2
        mse_test = (y_test_pred - y_test)**2

        nMAE_train = nMAE(y_train, y_train_pred)
        nMAE_test = nMAE(y_test, y_test_pred)

        mse = pd.concat([mse_train, mse_test]).reset_index(drop=False)
        #display(mse)
        plt.figure(figsize=(20, 6))
        relative_density_plot(mse['index'], mse['Power Generation(kW)'], num_bins=1000, show=False)
        plt.plot(mse['index'], mse['Power Generation(kW)'], color = 'green')
        plt.axvline(x=mse_train.shape[0])
        plt.legend()
        plt.show()

        #######
        power = pd.concat([y_train, y_test]).reset_index(drop=True)
        datetime = pd.concat([datetime_train, datetime_test]).reset_index(drop=True)
        plt.figure(figsize=(20,6))
        plt.plot(datetime, power, label='original',alpha=0.2)
        plt.plot(datetime, power.rolling(48).mean(), label='ma_48',alpha=0.7)
        plt.plot(datetime, power.rolling(150).mean(), label='ma_150',alpha=0.7)
        plt.plot(datetime, power.rolling(600).mean(), label='ma_600',alpha=0.7)
        plt.axvline(x=datetime_train.iloc[-1])
        plt.legend()
        plt.show()
        
#         print('train_r2_time :', train_r2[idx])
#         print('test_r2_time :', test_r2[idx])
#         print('train_nMAE_time :',nMAE(y_train, y_train_pred))
#         print('test_nMAE_time :',nMAE(y_test, y_test_pred))
        
    df_result['train r2 (time)'][0] = train_r2.mean()
    df_result['test r2 (time)'][0] = test_r2.mean()
    df_result['train r2 std (time)'][0] = train_r2.std()
    df_result['test r2 std (time)'][0] = test_r2.std()
    
    for idx in range(1, num_models+1) :
        df_result['train r2 (time)'][idx] = train_r2_models[idx-1].mean()
        df_result['test r2 (time)'][idx] = test_r2_models[idx-1].mean()
        df_result['train r2 std (time)'][idx] = train_r2_models[idx-1].std()
        df_result['test r2 std (time)'][idx] = test_r2_models[idx-1].std()
        
        
        mse_train = (y_train_pred - y_train)**2
        mse_test = (y_test_pred - y_test)**2

        nMAE_train = nMAE(y_train, y_train_hat)
        nMAE_test = nMAE(y_test, y_test_hat)

        mse = pd.concat([mse_train, mse_test]).reset_index(drop=False)
        #display(mse)
        plt.figure(figsize=(20, 6))
        relative_density_plot(mse['index'], mse['Power Generation(kW)'], num_bins=1000, show=False)
        plt.plot(mse['index'], mse['Power Generation(kW)'], color = 'green')
        plt.axvline(x=mse_train.shape[0])
        plt.legend()
        plt.show()

        #######
        power = pd.concat([y_train, y_test]).reset_index(drop=True)
        datetime = pd.concat([datetime_train, datetime_test]).reset_index(drop=True)
        plt.figure(figsize=(20,6))
        plt.plot(datetime, power, label='original',alpha=0.2)
        plt.plot(datetime, power.rolling(48).mean(), label='ma_48',alpha=0.7)
        plt.plot(datetime, power.rolling(150).mean(), label='ma_150',alpha=0.7)
        plt.plot(datetime, power.rolling(600).mean(), label='ma_600',alpha=0.7)
        plt.axvline(x=datetime_train.iloc[-1])
        plt.legend()
        plt.show()
    
    return df_result


def evaluate_idea(df, model_list) :
    # ensembled model
    model = EnsembledRegressor(model_list)
    names = [i[0] for i in model_list]
    num_models = len(model_list)
    
    # result dataframe
    df_result = pd.DataFrame(columns=['model',
                                     'train r2 (time)',
                                     'test r2 (time)',
                                     'train r2 (random)',
                                     'test r2 (random)',
                                     'train r2 std (time)',
                                     'test r2 std (time)',
                                     'train r2 std (random)',
                                     'test r2 std (random)'])
    df_result['model'] = df_result['model'].astype(str)
    df_result = df_result.append({'model' : 'average'}, ignore_index=True)
    for name in names :
        df_result = df_result.append({'model' : name}, ignore_index=True)
    
    # x, y
    df = df.reset_index(drop=True)
    x = df.drop(['datetime', 'date', 'Power Generation(kW)'], axis=1)
    #x = df.drop(['Power Generation(kW)'], axis=1)
    y = df['Power Generation(kW)']
    
    # KFold Validation
    n_splits = 5
    
    #################################################################
    #  Random Split
    kf = KFold(n_splits=n_splits, shuffle=True)
    ## average
    train_r2 = np.zeros(n_splits)
    test_r2 = np.zeros(n_splits)
    ## for each model
    train_r2_models = np.zeros((num_models, n_splits))
    test_r2_models = np.zeros((num_models, n_splits))
    
    
    for idx, (train_index, test_index) in enumerate(kf.split(x)) :

        # split data
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # train
        model.fit(x_train, y_train)
        
        # evaluate
        
        ## each model score
        y_train_pred_all = model.predict_all(x_train)
        y_test_pred_all = model.predict_all(x_test)
        train_predictions = [i[1] for i in y_train_pred_all]
        test_predictions = [i[1] for i in y_test_pred_all]
        for idx_model, train_prediction in enumerate(train_predictions) : 
            train_r2_models[idx_model][idx] = r2_score(y_train, train_prediction)
        for idx_model, test_prediction in enumerate(test_predictions) : 
            test_r2_models[idx_model][idx] = r2_score(y_test, test_prediction)

        ## average score
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        train_r2[idx] = r2_score(y_train, y_train_pred)
        test_r2[idx] = r2_score(y_test, y_test_pred)
        
    df_result['train r2 (random)'][0] = train_r2.mean()
    df_result['test r2 (random)'][0] = test_r2.mean()
    df_result['train r2 std (random)'][0] = train_r2.std()
    df_result['test r2 std (random)'][0] = test_r2.std()
    
    for idx in range(1, num_models+1) :
        df_result['train r2 (random)'][idx] = train_r2_models[idx-1].mean()
        df_result['test r2 (random)'][idx] = test_r2_models[idx-1].mean()
        df_result['train r2 std (random)'][idx] = train_r2_models[idx-1].std()
        df_result['test r2 std (random)'][idx] = test_r2_models[idx-1].std()
        
    #################################################################
    # Time Split
    kf = KFold(n_splits=n_splits, shuffle=False)
    ## average
    train_r2 = np.zeros(n_splits)
    test_r2 = np.zeros(n_splits)
    ## for each model
    train_r2_models = np.zeros((num_models, n_splits))
    test_r2_models = np.zeros((num_models, n_splits))
    
    
    for idx, (train_index, test_index) in enumerate(kf.split(x)) :

        # split data
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # train
        model.fit(x_train, y_train)
        
        # evaluate
        
        ## each model score
        y_train_pred_all = model.predict_all(x_train)
        y_test_pred_all = model.predict_all(x_test)
        train_predictions = [i[1] for i in y_train_pred_all]
        test_predictions = [i[1] for i in y_test_pred_all]
        for idx_model, train_prediction in enumerate(train_predictions) : 
            train_r2_models[idx_model][idx] = r2_score(y_train, train_prediction)
        for idx_model, test_prediction in enumerate(test_predictions) : 
            test_r2_models[idx_model][idx] = r2_score(y_test, test_prediction)

        ## average score
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        train_r2[idx] = r2_score(y_train, y_train_pred)
        test_r2[idx] = r2_score(y_test, y_test_pred)
        
    df_result['train r2 (time)'][0] = train_r2.mean()
    df_result['test r2 (time)'][0] = test_r2.mean()
    df_result['train r2 std (time)'][0] = train_r2.std()
    df_result['test r2 std (time)'][0] = test_r2.std()
    
    for idx in range(1, num_models+1) :
        df_result['train r2 (time)'][idx] = train_r2_models[idx-1].mean()
        df_result['test r2 (time)'][idx] = test_r2_models[idx-1].mean()
        df_result['train r2 std (time)'][idx] = train_r2_models[idx-1].std()
        df_result['test r2 std (time)'][idx] = test_r2_models[idx-1].std()
    
    return df_result
