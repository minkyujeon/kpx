import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from collections.abc import Iterable
from torch.utils.data.sampler import SubsetRandomSampler, Sampler

from preprocess.functions.date_inspector import load_files

class Dataset(Dataset):
    def __init__(self,data):
        self.len = len(data)
        
        data_y = data['Power Generation(kW)+0'].values
        data_x = data.drop(['Power Generation(kW)+0','Power Generation(kW)+1','Power Generation(kW)+2','datetime','date','date(forecast)','datetime(forecast)'],axis=1).values
        
        #df_y = df_y.reshape(-1,1)
        
        self.x_data = torch.from_numpy(data_x).float()
        self.y_data = torch.from_numpy(data_y).float()
        
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    #, self.df_date.iloc[index]
    
    def __len__(self):
        return self.len

class RNNDataset(Dataset):
    def __init__(self,x,y):
        self.len = len(x)
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
    
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
    
class DatasetManager:
    def __init__(self,data):
        self.data = data
        self.train, self.test = train_test_split(self.data, test_size=0.2)
        self.train_dataset = Dataset(self.train)
        self.test_dataset = Dataset(self.test)
        
        self.df_y = data['Power Generation(kW)+0']
        self.df_x = data.drop(['Power Generation(kW)+0','Power Generation(kW)+1','Power Generation(kW)+2','datetime','date','date(forecast)','datetime(forecast)'],axis=1)
        
    def get_loaders(self, batch_size):
        return [
            DataLoader(
                self.train_dataset, batch_size,
                shuffle=True,
                num_workers=4 
            ),
            DataLoader(
                self.test_dataset, batch_size,
                shuffle=False,
                num_workers=4 
            )
        ]