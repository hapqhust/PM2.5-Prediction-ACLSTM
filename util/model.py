import numpy as np 
import torch 
import numpy as np
import math
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

def save_config(config, log_dir):
	path = os.path.join(log_dir, "config.json")
	with open(path, 'a', encoding='utf-8') as f:
	    json.dump(config, f, ensure_ascii=False, indent=4)

def save_results(y_true, y_pred, log_dir, target_station):
	m_mae = mae(y_true, y_pred)
	m_mape = mape(y_true, y_pred)
	m_rmse = rmse(y_true, y_pred)
	m_mse = mse(y_true, y_pred)
	m_r2score = r2_score(y_true, y_pred)
	m_nse = nse(y_true, y_pred)
	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	results = [ dt_string, target_station, m_mae, m_mape, m_rmse, m_mse, m_r2score]
	path = os.path.join(log_dir, "metrics.csv")
	with open(path, 'a') as file:
		writer = csv.writer(file)
		writer.writerow(["date", "Target Station", "MAE", "MAPE", "RMSE", "MSE", "R2_score"])
		writer.writerow(results)

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def nse(y_true, y_pred):
    return (1-(np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)))

def mse(y_true, y_pred):
	return mean_squared_error(y_true, y_pred)

def visualize(y_true, y_pred, log_dir, name):
	plt.plot(y_pred, label='preds')
	plt.plot(y_true, label='gt')
	plt.legend()
	plt.savefig(os.path.join(log_dir, name))
	plt.close()

def generate_log_dir(args):
	log_dir = os.path.join("log", args.model)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	return log_dir

def load_optimizer(config, model):
    optimizer_type = config['train']['optimizer']
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    return optimizer

def save_checkpoint(model, optimizer, path):
    checkpoints = {
        "model_dict": model.state_dict(),
        "optimizer_dict": optimizer.state_dict(),
    }
    torch.save(checkpoints, path)

def load_model(model, checkpoint_path):
    return model.load_state_dict(torch.load(checkpoint_path)["model_dict"])