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

def get_poi_data_array(config):
    file_path = config['data']['data_dir']
    file_poi = file_path + 'poi/POI.csv'

    poi_df = pd.read_csv(file_poi)
    arr = poi_df.iloc[:,1:].astype(np.float32).values
    return arr 

def get_data_array(config):
    file_path = config['data_dir']

    file_gauge = file_path + 'gauges_processed/'
    file_location = file_path + 'location.csv' 
    nan_station = config['nan_station']

    # list_station = list(set([ stat.split('.csv')[0] for stat in os.listdir(file_gauge)]) - set(nan_station))
    list_station = [ stat.split('.csv')[0] for stat in os.listdir(file_gauge) if stat.split('.csv')[0] not in nan_station]
    # print(list_station)
    
    list_input_ft = config['input_features']

    location_df = pd.read_csv(file_location)
    scaler = MinMaxScaler()
    location_ = []
    list_arr = []

    for stat in list_station:
      row_stat = location_df[location_df['location'] == stat] # name, lat, lon
      location_it = row_stat.values[:, [0,2,1]]
      location_.append(location_it)

      df = pd.read_csv(file_gauge  + f"{stat}.csv")
      df_ = df[list_input_ft]
      arr = df_.iloc[:,:].astype(np.float32).values
      list_arr.append(arr)

    num_ft = list_arr[0].shape[-1]  #14
    list_arr = np.concatenate(list_arr, axis=0) # 8642 * 20, 14
    
    scaler.fit(list_arr)
    transformed = scaler.transform(list_arr)
    # print(transformed.shape)

    # transformed = list_arr.copy()

    transformed_data  =  transformed.reshape( len(list_station), -1 , num_ft) # 33, 8642, 14
    transformed_data = np.swapaxes(transformed_data, 0,1 ) # 8642, 33, 14 

    location_ = np.concatenate(location_, axis=0)
    return transformed_data,  location_, list_station, scaler

if __name__ == '__main__':
    config_file = './config/lstm.yml'
    with open(config_file) as f:
      conf = yaml.safe_load(f)

    arr = get_poi_data_array(conf)
    print(arr.shape)