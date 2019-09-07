import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import math
pd.options.mode.chained_assignment = None
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
# regression models

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from preprocess.functions.date_inspector import load_files
from eda.functions.eda import show_relative_density_plot, relative_density_plot
from functions.evaluate import EnsembledRegressor, evaluate_idea, evaluate_idea_2, nMAE
from functions.feature_engineering import wind_cos_sin, new_wind_speed_direction, make_moving_average_df, load_power_ma_forecast
from functions.feature_engineering import load_power_ma_forecast_mean, fe_add_timestep, fe_add_previous_n_hours_mean_kpx, add_time_feature

class StackedRegressor() :
    def __init__(self) :
        self.model_y1_xgb = XGBRegressor(nthread=16, verbose=0, objective='reg:squarederror')
        self.model_y1_cat = CatBoostRegressor(thread_count=16, verbose=0)
        self.model_y1_lgbm = LGBMRegressor(num_threads=16, verbose=0)
        self.model_y2_xgb = XGBRegressor(nthread=16, verbose=0, objective='reg:squarederror')
        self.model_y2_cat = CatBoostRegressor(thread_count=16, verbose=0)
        self.model_y2_lgbm = LGBMRegressor(num_threads=16, verbose=0)
        self.model_y3_xgb = XGBRegressor(nthread=16, verbose=0, objective='reg:squarederror')
        self.model_y3_cat = CatBoostRegressor(thread_count=16, verbose=0)
        self.model_y3_lgbm = LGBMRegressor(num_threads=16, verbose=0)
        self.model_y1_stack = XGBRegressor(nthread=16, verbose=0, objective='reg:squarederror')
        self.model_y2_stack = XGBRegressor(nthread=16, verbose=0, objective='reg:squarederror')
        self.model_y3_stack = XGBRegressor(nthread=16, verbose=0, objective='reg:squarederror')
    def fit(self, x, y1, y2, y3) :
        self.model_y1_xgb.fit(x, y1)
        self.model_y1_cat.fit(x, y1)
        self.model_y1_lgbm.fit(x, y1)
        y1_hat_xgb = self.model_y1_xgb.predict(x)
        y1_hat_cat = self.model_y1_cat.predict(x)
        y1_hat_lgbm = self.model_y1_lgbm.predict(x)
        x1 = x.copy()
        x1['xgb']= y1_hat_xgb
        x1['cat'] = y1_hat_cat
        x1['lgbm'] = y1_hat_lgbm
        self.model_y1_stack.fit(x1, y1)
        self.model_y2_xgb.fit(x, y2)
        self.model_y2_cat.fit(x, y2)
        self.model_y2_lgbm.fit(x, y2)
        y2_hat_xgb = self.model_y2_xgb.predict(x)
        y2_hat_cat = self.model_y2_cat.predict(x)
        y2_hat_lgbm = self.model_y2_lgbm.predict(x)
        x2 = x.copy()
        x2['xgb']= y2_hat_xgb
        x2['cat'] = y2_hat_cat
        x2['lgbm'] = y2_hat_lgbm
        self.model_y2_stack.fit(x2, y2)
        self.model_y3_xgb.fit(x, y3)
        self.model_y3_cat.fit(x, y3)
        self.model_y3_lgbm.fit(x, y3)
        y3_hat_xgb = self.model_y3_xgb.predict(x)
        y3_hat_cat = self.model_y3_cat.predict(x)
        y3_hat_lgbm = self.model_y3_lgbm.predict(x)
        x3 = x.copy()
        x3['xgb']= y3_hat_xgb
        x3['cat'] = y3_hat_cat
        x3['lgbm'] = y3_hat_lgbm
        self.model_y3_stack.fit(x3, y3)
    def predict(self, x) :
        y1_hat_xgb = self.model_y1_xgb.predict(x)
        y1_hat_cat = self.model_y1_cat.predict(x)
        y1_hat_lgbm = self.model_y1_lgbm.predict(x)
        x1 = x.copy()
        x1['xgb']= y1_hat_xgb
        x1['cat'] = y1_hat_cat
        x1['lgbm'] = y1_hat_lgbm
        y1_hat = self.model_y1_stack.predict(x1)
        y2_hat_xgb = self.model_y2_xgb.predict(x)
        y2_hat_cat = self.model_y2_cat.predict(x)
        y2_hat_lgbm = self.model_y2_lgbm.predict(x)
        x2 = x.copy()
        x2['xgb']= y2_hat_xgb
        x2['cat'] = y2_hat_cat
        x2['lgbm'] = y2_hat_lgbm
        y2_hat = self.model_y2_stack.predict(x2)
        y3_hat_xgb = self.model_y3_xgb.predict(x)
        y3_hat_cat = self.model_y3_cat.predict(x)
        y3_hat_lgbm = self.model_y3_lgbm.predict(x)
        x3 = x.copy()
        x3['xgb']= y3_hat_xgb
        x3['cat'] = y3_hat_cat
        x3['lgbm'] = y3_hat_lgbm
        y3_hat = self.model_y3_stack.predict(x3)
        return y1_hat, y2_hat, y3_hat