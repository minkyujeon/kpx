#날씨 예보 데이터 전처리

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
pd.options.mode.chained_assignment = None

def interval6_to_interval3(x) :
    return 2*x -1

def preprocess_hour_interval(file_dir, year, hour=3) : 
        
    df = pd.read_csv(file_dir)
    
    # hour가 200.0, 2300.0과 같이 되어있음
    df['hour'] = df['hour']/100
    
    # df month 생성
#     print('df_shape:',df.shape[0])
#     if '단기예보_제주도_제주시_한경면_2014_6시간강수량.csv' in file_dir:
#         #df.drop([108039],axis=0).tail()
#         df = df.drop([len(df)-1],axis=0)
#         print('len(df)',len(df))
#         display(df.tail())

    df['month'] = np.zeros(df.shape[0])
    # 각 월이 시작하는걸 알려주는 row는 값이 비어있음
    indices_month = df[df['hour'].isna() == True].index
    
    if indices_month.shape[0] == 11 :
        df['month'][:indices_month[0]] = 1
        df['month'][indices_month[0]:indices_month[1]] = 2
        df['month'][indices_month[1]:indices_month[2]] = 3
        df['month'][indices_month[2]:indices_month[3]] = 4
        df['month'][indices_month[3]:indices_month[4]] = 5
        df['month'][indices_month[4]:indices_month[5]] = 6
        df['month'][indices_month[5]:indices_month[6]] = 7
        df['month'][indices_month[6]:indices_month[7]] = 8
        df['month'][indices_month[7]:indices_month[8]] = 9
        df['month'][indices_month[8]:indices_month[9]] = 10
        df['month'][indices_month[9]:indices_month[10]] = 11
        df['month'][indices_month[10]:] = 12
    else :
        num_months = indices_month.shape[0] # get number of months
        df['month'][:indices_month[0]] = 1
        print('num_months:',num_months)
        for idx in range (1, num_months) :
#             print('idx : ', idx)
#             print('indieces_month[idx-1]:',indices_month[idx-1])
#             print('indieces_month[idx]:',indices_month[idx])
            df['month'][indices_month[idx-1]:indices_month[idx]] = idx+1
            
        df['month'][indices_month[num_months-1]:] = num_months+1
    

    # 각 월이 시작하는것을 알려주는 row 삭제 (na값)
    df = df.drop(indices_month, axis=0)

    # 6시간 단위인 경우
    if hour == 6 :
        df['forecast'] = df['forecast'].apply(interval6_to_interval3)
    # 3시간 단위인 경우 아무 처리도 하지 않음
    
    # 자료형 변경
    df['hour'] = df['hour'].astype('int64')
    df['month'] = df['month'].astype('int64')
    df[df.columns[0]] = df[df.columns[0]].astype('int64')
    df['date'] = np.zeros(df.shape[0])
    df['date'] = df['date'].astype('object')
    df['datetime'] = np.zeros(df.shape[0])
    df['datetime'] = df['datetime'].astype('object')
    df['datetime(forecast)'] = np.zeros(df.shape[0])
    df['datetime(forecast)'] = df['datetime(forecast)'].astype('object')

    # datetime.date 생성
    year = year
    df['date'] = df.apply(lambda row : datetime.date(year, row['month'], row[df.columns[0]]), axis=1)

    # datetime.datetime (현재 시간) 생성
    df['datetime'] = df.apply(lambda row : datetime.datetime(year, row['month'], row[df.columns[0]], row['hour'], 0, 0), axis=1)

    # datetime.datetime (예보가 가리키는 미래의 시간) 생성
    df['datetime(forecast)'] = df.apply(lambda row : row['datetime']+ datetime.timedelta(hours=row['forecast']), axis=1)
    
    # datetime.date (예보가 가리키는 미래의 날짜) 생성
    df['date(forecast)'] = df['datetime(forecast)'].dt.date

    df = df.drop(df.columns[0:3], axis=1)
    df = df.drop(['month'], axis=1)

    df = df.rename(columns={df.columns[0] : 'value'})
    
    return df

def get_df_temp(filename, year, hour) :
    year_idx = filename.index(str(year)+'_')
    column_name = filename[year_idx+5:-4]
    file_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'raw', 'kma', filename))
    df_temp = preprocess_hour_interval(file_dir, year, hour=hour)
    df_temp = df_temp.rename(columns={'value':column_name})
    
    return df_temp, year_idx, column_name

def merge_df(df, df_temp, first_column) :
    df = pd.merge(df, df_temp, how='outer', on=['datetime', 'datetime(forecast)'])
    df = df[df[first_column].notnull()]
    df = df.rename(columns={'date_x':'date'})
    df = df.drop(['date_y'], axis=1)
    df = df.rename(columns={'date(forecast)_x':'date(forecast)'})
    df = df.drop(['date(forecast)_y'], axis=1)
    return df
    

def preprocess_forecast(data_dir, location_year, location) :
    
    filelist_all = os.listdir(data_dir)
    filelist_forecast = [s for s in filelist_all if '단기예보' in s]
    first_column = ''
    #filelist_solar = [s for s in filelist_all if '일사량' in s]
    
    year = int(location_year[-4:])
    print('year : ', year)
    
    fl = [s for s in filelist_forecast if location_year in s]
    fl24 = [s for s in fl if '일' in s]
    fl6 = [s for s in fl if '6시간' in s]
    fl3 = list(set(fl)-set(fl6)-set(fl24))
    
    print('fl24 : ', fl24)
    print('fl6 : ', fl6)
    print('fl3 : ', fl3)
    
    # initial df
    df = pd.DataFrame(columns=['date', 'date(forecast)', 'datetime', 'datetime(forecast)', 'location'])
        
    # 3시간단위로 나오는 데이터
    for idx, file3 in enumerate(fl3) :
        print('file3 : ', file3)
        df_temp, year_idx, column_name = get_df_temp(file3, year, 3)
        
        # initial state
        if idx == 0 :
            first_column = column_name
            df['date'] = df_temp['date']
            df['datetime'] = df_temp['datetime']
            df['date(forecast)'] = df_temp['date(forecast)']
            df['datetime(forecast)'] = df_temp['datetime(forecast)']
            df[column_name] = df_temp[column_name]
        else :
            df[column_name] = df_temp[column_name]
    
    # 6시간단위로 나오는 데이터
    for file6 in fl6 :
        print('file6 : ', file6)
        df_temp, year_idx, column_name = get_df_temp(file6, year, 6)
        
        df = merge_df(df, df_temp, first_column)
        
    # 24시간단위로 나오는 데이터
    for file24 in fl24 :
        print('file24 : ', file24)
        df_temp, year_idx, column_name = get_df_temp(file24, year, 3)
        
        df = merge_df(df, df_temp, first_column)
        
    df['location'] = df['location'].apply(lambda x : location)
        
    # return
    return df
    
