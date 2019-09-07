import math
import datetime
import pandas as pd
import numpy as np
import os

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


#풍향/풍속 인코딩
def wind_cos_sin(df):
    wind_dir = df['Wind Direction(16)']
    wind_dir_deg = np.deg2rad(wind_dir)

    wind_dir_cos = wind_dir_deg.apply(math.cos)
    wind_dir_sin = wind_dir_deg.apply(math.sin)

    df['wind_dir_cos'] = wind_dir_cos#.round(1)
    df['wind_dir_sin'] = wind_dir_sin#.round(1)
    
    return df

def new_wind_speed_direction(df,phi):
    theta = df['Wind Direction(16)']
    wind_speed = df['Wind Speed(m/s)']
    deg = theta - phi

    cos_deg = np.deg2rad(deg).apply(math.cos)

    new_wind_speed = wind_speed*cos_deg

    df['new_wind_speed'] = new_wind_speed#.round(1)
    
    return df

#발전량 moving average
def make_moving_average_df(df,hours,year):
    df = df.reset_index(drop=True)
    if year == 0:
        df_cycle = pd.DataFrame(columns=['datetime'])
        df_cycle['datetime'] = df['datetime'].loc[:(24)*365-1]
        for hour in hours:
            name = 'ma'+str(hour)

            df_cycle[name]= df['발전량(kW)'].rolling(hour).mean().shift(-hour).loc[year:(year+1)*(24)*365-1]
    
    elif year == 1:
        df_cycle = pd.DataFrame(columns=['datetime'])
        df_cycle['datetime'] = df['datetime'].loc[year*(24)*365:(year+1)*(24)*365-1]
        for hour in hours:
            name = 'ma'+str(hour)

            df_cycle[name]= df['발전량(kW)'].rolling(hour).mean().shift(-hour).loc[24*365:2*(24)*365-1]
    
            
    return df_cycle.reset_index(drop=True)

#발전량 moving average feature추가 
def load_power_ma_forecast(df_target_date, df_ma, hour):
    
    if df_target_date.month >= 7 :
        year = 2017
    elif df_target_date.month < 7 :
        year = 2018
    
    target = datetime.datetime(year,
                               df_target_date.month, 
                               df_target_date.day, 
                               df_target_date.hour,
                              0,
                              0)
    
    
    name = 'ma'+str(hour)+'_'+str(year)
    
    try:
        return float(df_ma[df_ma['datetime'] == target][name])
    except Exception as e :
        return -1

def load_power_ma_forecast_mean(df_target_date, df_ma, hour):
    
    year = 2017
    
    target = datetime.datetime(year,
                               df_target_date.month, 
                               df_target_date.day, 
                               df_target_date.hour,
                              0,
                              0)
    
    name = 'ma'+str(hour)+'_mean'
    try:
        return float(df_ma[df_ma['datetime'] == target][name])

    except Exception as e :
        return -1

    
# Feature Windowing
def fe_add_timestep(df_original, num_timestep) : 
    
    num_timestep = num_timestep//3
    df = df_original.copy()
    df_shifted = df_original.copy()
    df_shifted = df_shifted.shift(num_timestep)

    ### previous, later 함수 대상으로는 돌지 않게
    lst = list(df.columns)
    lst2 = list(df.columns)
    for col in lst:
        if 'previous' in col :
            lst2.remove(col)
    for col in lst:
        if 'later' in col :
            lst2.remove(col)
    columns = lst2
    columns.remove('date')
    columns.remove('date(forecast)')
    columns.remove('datetime')
    columns.remove('datetime(forecast)')
#     columns.remove('Power Generation(kW)+0')
#     columns.remove('Power Generation(kW)+1')
#     columns.remove('Power Generation(kW)+2')
    columns.remove('location')
    
    num_timestep = num_timestep*3
    for column in columns :
        df[column+' (previous %d)'%num_timestep] = df_shifted[column]
    df = df[df['Celsius(Lowest) (previous %d)'%num_timestep].notnull()]
    
    return df

#add feature moving average
def fe_add_previous_n_hours_mean_kpx(df_original, columns, how_long=1):
    df = df_original.copy()
    
    a = df['datetime(forecast)'].loc[0]
    
    n = df[df['datetime(forecast)'] == datetime.datetime(a.year, a.month, a.day, a.hour+15)].index[0]-1
    how_long = how_long*n
    
    for column in columns :
        df[column+'('+str(n*12)+' hours mean)'] = 0
        for idx in range(how_long) :
            df[column+'('+str(n*12)+'hours mean)'] += df[column].shift(idx+1)
        df[column+'('+str(n*12)+' hours mean)'] /= how_long

        df[column+'('+str(n*12)+'hours mean)'] = df[column+'('+str(n*12)+'hours mean)'].astype(float).round(1)
    df = df[how_long:]    

    return df

def add_time_feature(df):
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    
    return df