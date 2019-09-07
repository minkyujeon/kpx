#날씨 예보, 발전량 데이터 불러오기

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import math
from ipywidgets import interact, interact_manual
pd.options.mode.chained_assignment = None

def load_files(observation_list, forecast_list, filename_power) : 
    for idx, filename_observation in enumerate(observation_list) :
        if idx == 0:
            df_observation = pd.read_pickle(os.path.abspath(os.path.join(os.getcwd(), '..', 'data', filename_observation)))
        else :
            df_temp = pd.read_pickle(os.path.abspath(os.path.join(os.getcwd(), '..', 'data', filename_observation)))
            df_observation = pd.concat([df_observation, df_temp], axis=0)
            
    for idx, filename_forecast in enumerate(forecast_list) :
        if idx == 0 :
            df_forecast = pd.read_pickle(os.path.abspath(os.path.join(os.getcwd(), '..', 'data', filename_forecast)))
        else :
            df_temp = pd.read_pickle(os.path.abspath(os.path.join(os.getcwd(), '..', 'data', filename_forecast)))
            df_forecast= pd.concat([df_forecast, df_temp], axis=0)

    df_power = pd.read_pickle(os.path.abspath(os.path.join(os.getcwd(), '..', 'data', filename_power)))
    
    df_forecast = df_forecast.reindex(columns=['date',
                                               'date(forecast)',
                                               'datetime',
                                               'datetime(forecast)',
                                               'location',
                                               '하늘상태',
                                               '풍속',
                                               '습도',
                                               '3시간기온',
                                               '풍향',
                                               '강수형태',
                                               '강수확률',
                                               '6시간강수량',
                                               '6시간적설',
                                               '일최고기온',
                                               '일최저기온'])

    return df_observation, df_forecast, df_power

def load_files_legacy(observation_list, forecast_list, solar_list, filename_power) :
    for idx, filename_observation in enumerate(observation_list) :
        if idx == 0:
            df_observation = pd.read_pickle(os.path.abspath(os.path.join(os.getcwd(), '..', 'data', filename_observation)))
        else :
            df_temp = pd.read_pickle(os.path.abspath(os.path.join(os.getcwd(), '..', 'data', filename_observation)))
            df_observation = pd.concat([df_observation, df_temp], axis=0)


    for idx, filename_forecast in enumerate(forecast_list) :
        if idx == 0 :
            df_forecast = pd.read_pickle(os.path.abspath(os.path.join(os.getcwd(), '..', 'data', filename_forecast)))
        else :
            df_temp = pd.read_pickle(os.path.abspath(os.path.join(os.getcwd(), '..', 'data', filename_forecast)))
            df_forecast= pd.concat([df_forecast, df_temp], axis=0)

    for idx, filename_solar in enumerate(solar_list) :
        if idx == 0:
            df_solar = pd.read_pickle(os.path.abspath(os.path.join(os.getcwd(), '..', 'data', filename_solar)))
        else :
            df_temp = pd.read_pickle(os.path.abspath(os.path.join(os.getcwd(), '..', 'data', filename_solar)))
            df_solar = pd.concat([df_solar, df_temp], axis=0)

    df_power = pd.read_pickle(os.path.abspath(os.path.join(os.getcwd(), '..', 'data', filename_power)))

    df_forecast = df_forecast.reindex(columns=['date',
                                               'date(forecast)',
                                               'datetime',
                                               'datetime(forecast)',
                                               'location',
                                               '하늘상태',
                                               '풍속',
                                               '습도',
                                               '3시간기온',
                                               '풍향',
                                               '강수형태',
                                               '강수확률',
                                               '6시간강수량',
                                               '6시간적설',
                                               '일최고기온',
                                               '일최저기온'])

    return df_observation, df_forecast, df_solar, df_power


def inspect_day(date_target, df_observation, df_forecast, df_solar, df_power) : 
    # solar radiation and power generation
    titlesize = 16
    fontsize = 14

    fig, ax1 = plt.subplots(figsize=(16, 3))
    fig.suptitle('Solar Radiation / Power Generation', fontsize=titlesize)

    color = 'tab:red'
    ax1.set_xlabel('timestamp', fontsize=fontsize)
    ax1.set_ylabel('power generation', color=color, fontsize=fontsize)
    ax1.plot(df_power[df_power['date']==date_target]['datetime'], df_power[df_power['date']==date_target]['발전량(kW)'], color=color, label='power generation')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('solar radiation', color=color, fontsize=fontsize)  # we already handled the x-label with ax1
    ax2.plot(df_solar[df_solar['date']==date_target]['datetime'], df_solar[df_solar['date']==date_target]['일사량(MJ/m^2)'], color=color, label='solar radiation')
    ax2.tick_params(axis='y', labelcolor=color)

    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend()
    plt.show()
    
    # celsius
    plt.figure(figsize=(15.8, 2))
    plt.plot(df_solar[df_observation['date']==date_target]['datetime'], df_observation[df_observation['date']==date_target]['기온(°C)'],  label='celsius', color='plum')
    plt.title('Celsius', fontsize=titlesize)
    plt.legend()
    plt.xlabel('timestamp', fontsize=fontsize)
    plt.ylabel('celsius', fontsize=fontsize)
    plt.show()

    # observation
    rainfall_global_max = df_observation['강수량(mm)'].max()
    snowfall_global_max = df_observation['적설(cm)'].max()
    cloud_global_max = df_observation['전운량(10분위)'].max()
    rainfall_target_max = df_observation[df_observation['date']==date_target]['강수량(mm)'].max()
    rainfall_target_min = df_observation[df_observation['date']==date_target]['강수량(mm)'].min()
    snowfall_target_max = df_observation[df_observation['date']==date_target]['적설(cm)'].max()
    snowfall_target_min = df_observation[df_observation['date']==date_target]['적설(cm)'].min()
    cloud_target_max = df_observation[df_observation['date']==date_target]['전운량(10분위)'].max()
    cloud_target_min = df_observation[df_observation['date']==date_target]['전운량(10분위)'].min()

    fig, ax1 = plt.subplots(figsize=(16, 3))
    fig.suptitle('Weather Observation', fontsize=titlesize)

    color = 'tab:orange'
    ax1.set_xlabel('timestamp', fontsize=fontsize)
    ax1.set_ylabel('rainfall', color=color, fontsize=fontsize)
    ax1.plot(df_observation[df_observation['date']==date_target]['datetime'], df_observation[df_observation['date']==date_target]['강수량(mm)'], color=color, label='rainfall')
    if rainfall_target_max == 0 : 
        ax1.set_ylim([-0.2, cloud_global_max])
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color1 = 'tab:green'
    color2 = 'tab:purple'
    ax2.set_ylabel('cloud, snowfall', color=color1, fontsize=fontsize)  # we already handled the x-label with ax1
    ax2.plot(df_observation[df_observation['date']==date_target]['datetime'], df_observation[df_observation['date']==date_target]['전운량(10분위)'], color=color1, label='cloud')
    ax2.plot(df_observation[df_observation['date']==date_target]['datetime'], df_observation[df_observation['date']==date_target]['적설(cm)'], color=color2, label='snowfall')
    if cloud_target_max == 0 : 
        ax2.set_ylim([-0.2, cloud_global_max])
    ax2.tick_params(axis='y', labelcolor=color1)

    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend()
    plt.show()

    
    
    print('해당 날짜 rainfall\t최소값: %.2f, 최대값 : %.2f, 최대값(전구간) : %.2f' % (rainfall_target_min, rainfall_target_max, rainfall_global_max))
    print('해당 날짜 snowfall\t최소값: %.2f, 최대값 : %.2f, 최대값(전구간) : %.2f' %(snowfall_target_min, snowfall_target_max, snowfall_global_max))
    print('해당 날짜 cloud\t\t최소값: %.2f, 최대값 : %.2f, 최대값(전구간) : %.2f' %(cloud_target_min, cloud_target_max, cloud_global_max))

    display(df_observation[df_observation['date']==date_target].drop(['date', 'location'], axis=1))
    
    

    return
