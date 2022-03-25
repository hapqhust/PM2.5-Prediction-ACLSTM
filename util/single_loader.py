'''
    Dataloader for multiple stations 
    get_data_array: get minmax scaled data in numpy form, location numpy array, list of station names,  scaler
    get_distance: distance between 2 coordinations
'''
import yaml 

import os 
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import yaml
import os 

def get_data_array(config, target_station):
    file_path = config['data_dir']

    file_gauge = file_path + 'gauges_processed/'

    list_input_ft = config['input_features']
    scaler = MinMaxScaler()

    df = pd.read_csv(file_gauge  + f"{target_station}.csv")[list_input_ft]
    arr = df.iloc[:,:].astype(np.float32).values

    scaler.fit(arr)
    transformed_data = scaler.transform(arr)
    return transformed_data,  scaler

if __name__ == '__main__':
    config_path = './config/lstm.yml'
    with open(config_path, 'r') as f:
        config= yaml.safe_load(f)
    data, scaler = get_data_array(config) 
    print(data.shape)