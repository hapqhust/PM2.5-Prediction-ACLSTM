import pandas as pd 
import numpy as np 

from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
import yaml 

import os 
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import yaml
import os 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from utils import get_data_array, get_dataloader 

def get_config(model_type):
    config_path = './config/'
    return config_path + model_type + '.yml'

def to_numeric(x):
    x_1 = x.apply(pd.to_numeric, errors="coerce")
    res = x_1.clip(lower=0)
    return res

def fill_na(x):
    res= x.ffill()
    res.fillna(0, inplace=True)
    return res 

def clipping(x):
    min_clip = 0.0
    max_clip = 0.95
    ans = x.transform(lambda x: np.clip(x, x.quantile(min_clip), x.quantile(max_clip) ) )
    return ans

def preprocess_pipeline(df, type):
    lst_cols = list(set(list(df.columns)) - set(['Hour','Day','Month','Year']))
    type_transformer = FunctionTransformer(to_numeric)
    fill_transformer = FunctionTransformer(fill_na)
    clipping_transformer = FunctionTransformer(clipping)
    num_pl = Pipeline(
        steps=[
            ("fill_na", fill_transformer),
            ("numeric_transform", type_transformer),
            ("clipping", clipping_transformer),
        ],
    )

    preprocessor = ColumnTransformer(transformers=[("num", num_pl, lst_cols)])
    res = preprocessor.fit_transform(df)
 
    trans_df = pd.DataFrame(res, columns=lst_cols)
    trans_df[['Hour','Day','Month','Year']] = df[['Hour','Day','Month','Year']]
    
    lst_meteo_cols = ['PM2.5', 'Mean','AQI','PM10', 'CO', 'NO2', 'O3', 'SO2', 'prec', 'lrad', 'shum', 'pres', 'temp', 'wind', 'srad']

    final_lst_cols  = ['Hour','Day','Month','Year'] + lst_meteo_cols
    trans_df = trans_df[final_lst_cols]
    trans_df.reset_index(drop=True, inplace=True)
    return trans_df

if __name__=='__main__':
    model = 'spatio_attention_embedded_rnn'

    data_folder =  './data/Beijing/gauges/'

    out_folder =  './data/Beijing/gauges_processed/'

    lst_stations = os.listdir(data_folder)

    for stat in lst_stations:
        df = pd.read_csv(data_folder + stat, index_col=0)
        trans_df = preprocess_pipeline(df, model)
        trans_df.to_csv(out_folder + stat, index=False)
