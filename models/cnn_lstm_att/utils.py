import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import os
import glob

def get_dataloader(dataset_class, batch_size, shuffle=True):
    return DataLoader(
        dataset_class, 
        batch_size=batch_size, 
        shuffle=shuffle
    )

class Generator(Dataset):
    def __init__(self, inp, out):
        self.inp = inp
        self.out = out
    def __len__(self):
        return len(self.inp)
    def __getitem__(self, index):
        return self.inp[index], self.out[index]

def get_data(folder, target_stations, station_name):
    #Dataframe lưu giá trị của tất cả các trạm
    df_aqme = {}

    #Danh sách tên tất cả các trạm
    stations = target_stations
    
    index = stations.index(station_name)
    print(index)
    
    for station in stations:
        if station != '':
            file_name = folder + "{}.csv".format(station)
            #if file_name.split('.')[0] != 'location':
            # import pdb; pdb.set_trace()

            df_aqme[station] = pd.read_csv(file_name)
            #stations.append(file_name.split('.')[0])
    print(stations)

    # Chọn trạm đầu tiên để dự đoán
    df_aqme1 = df_aqme[stations[index]]
    df_aqme1 = df_aqme1.iloc[:,2:]


    #Lấy chỉ số PM2.5 của tất cả các trạm và đánh giá hệ số tương quan
    pm2_5_all = pd.DataFrame()
    for station in stations:
        if station != '':
            pm2_5_all[station] = df_aqme[station]['PM2_5']    
    pm2_5_all.columns = [str(i) for i in range(0, len(pm2_5_all.columns))]
    station_corr = pm2_5_all.corr(method = 'pearson')

    #Chọn k trạm có độ tương quan cao nhất 
    data = station_corr.sort_values(by=str(index), ascending=False).index[1:1].to_numpy() 

    #print(station_name)
    for i in data:
        s = 'PM2_5_S' + i
        #print(s)
        df_aqme1[s] = df_aqme[stations[int(i)]]["PM2_5"]
    print(df_aqme1.columns)
    return df_aqme1

def preprocess(config, nan_stations, station_name):
    folder = config["dataset_dir"]
    seq_len = config["input_len"]
    horizon = config["horizon"]
    
    target_stations = []

    for path in glob.glob(os.path.join(folder, '*.csv')):
        file_name = os.path.basename(path)
        station_name = file_name.split('.')[0]
        if station_name not in nan_stations: 
            target_stations.append(station_name)

    train_size = config['train_size']
    valid_size = config['valid_size']

    df_aqme1 = get_data(folder, target_stations, station_name)
    X = df_aqme1
    y = df_aqme1[['PM2_5']]
    # import pdb; pdb.set_trace()

    # Xử lý outliers
    #X = remove_outlier(X)
    #y = remove_outlier(y)

    # Phân chia tập train, test, validation
    X_train = X[:int(len(X)*train_size)]
    X_val = X[int(len(X)*train_size):int(len(X)*(train_size + valid_size))]
    X_test = X[int(len(X)*(train_size + valid_size)):]

    y_train = y[:int(len(y)*train_size)]
    y_val = y[int(len(y)*train_size):int(len(y)*(train_size + valid_size))]
    y_test = y[int(len(y)*(train_size + valid_size)):]

    # Dùng MinMaxScaler
    scale_X = MinMaxScaler()
    scale_y = MinMaxScaler()

    scale_X.fit(X_train)
    scale_y.fit(y_train)

    X_train_data = scale_X.transform(X_train)
    X_test_data = scale_X.transform(X_test)
    X_valid_data = scale_X.transform(X_val)
    y_valid_data = scale_y.transform(y_val)
    y_train_data = scale_y.transform(y_train)
    y_test_data = scale_y.transform(y_test)


    X_train = np.array([X_train_data[i:i+seq_len] for i in range(0, len(X_train_data) - seq_len - horizon)])
    X_test = np.array([X_test_data[i:i+seq_len] for i in range(0, len(X_test_data) - seq_len - horizon)])
    X_valid = np.array([X_valid_data[i:i+seq_len] for i in range(0, len(X_valid_data) - seq_len - horizon)])

    y_train = np.array([y_train_data[i+seq_len: i+seq_len+horizon] for i in range(0, len(y_train_data) - seq_len - horizon)])
    y_test = np.array([y_test_data[i+seq_len: i+seq_len+horizon] for i in range(0, len(y_test_data) - seq_len - horizon)])
    y_valid = np.array([y_valid_data[i+seq_len: i+seq_len+horizon] for i in range(0, len(y_valid_data) - seq_len - horizon)])

    # Chuyển sang tensor
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    X_valid = torch.FloatTensor(X_valid)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    y_valid = torch.FloatTensor(y_valid)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, scale_y


