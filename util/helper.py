import numpy as np
import torch

from models.multitask_lstm_autoenc.supervisor import MultitaskLSTMAutoencSupervisor
from models.lstm.supervisor import LSTMSupervisor
from models.spatio_attention_embedded_rnn.supervisor import SAERSupervisor
from models.spatio_attention_embedded_rnn.supervisor import SpatioEmbeddedRNN
from models.cnn_lstm_att.supervisor import CNN_LSTM_ATTSupervisor
from models.encoder_decoder_lstm.supervisor import EDLSTMSupervisor
def get_config(model_type):
    config_path = 'config/'
    return config_path + model_type + '.yml'

def model_mapping(model_type):
    config_path = 'config/'
    if model_type == 'multitask_lstm_autoenc':
        res = {
            'model': MultitaskLSTMAutoencSupervisor,
            'config': config_path + model_type + '.yml'
        }
    elif model_type == 'lstm':
        res = {
            'model': LSTMSupervisor,
            'config': config_path + model_type + '.yml'
        }
    elif model_type == 'spatio_attention_embedded_rnn':
        res = {
            'model': SAERSupervisor,
            'config': config_path + model_type + '.yml'
        }
    elif model_type == 'encoder_decoder_lstm':
        res = {
            'model': EDLSTMSupervisor,
            'config': config_path + model_type + '.yml'
        }
    elif model_type == 'cnn_lstm_att':
        res = {
            'model': CNN_LSTM_ATTSupervisor,
            'config': config_path + model_type + '.yml'
        }
    
    return res 

def seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
