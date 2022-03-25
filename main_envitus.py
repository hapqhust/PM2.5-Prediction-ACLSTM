from email.policy import default
import argparse
import torch 
import yaml
import wandb

from util.helper import seed, model_mapping
from util.wb_loader import get_wandb_instance

# them tat ca cac tham so muon chinh vao day
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        type=str,
                        default='train')
    parser.add_argument('--seed',
                        default=52,
                        type=int,
                        help='Seed')
    parser.add_argument('--model',
                        default='cnn_lstm_att',
                        type=str)
    parser.add_argument('--batch_size',
                        type=int)
    parser.add_argument('--hidden_size',
                        type=int)
    parser.add_argument('--dropout',
                        type=float)                        
    parser.add_argument('--lr',
                        type=float)
    parser.add_argument('--lr_decay_ratio',
                        type=float)
    return parser

if __name__=="__main__":
    parser = parse_args()
    args = parser.parse_args()

    seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = model_mapping(args.model)
    with open(conf['config']) as f:
        config = yaml.safe_load(f)

    target_station = config['data']['target_station']
    config_wandb = {
        #'target_station': config['data']['target_station'],
        'dataset_dir': config['data']['dataset_dir'],
        'batch_size': config['data']['batch_size'],
        'train_size': config['data']['train_size'],
        'valid_size': config['data']['valid_size'],
        'input_features': config['model']['input_features'],
        'target_features': config['model']['target_features'],
        'input_len': config['model']['input_len'],
        'horizon': config['model']['horizon'],
        'kernel_size': config['model']['kernel_size'],
        'hidden_size': config['model']['hidden_size'],
        'num_layers': config['model']['num_layers'],
        'dropout': config['model']['dropout'],
        'epochs': config["train"]["epochs"],
        'optimizer': config['train']['optimizer'],
        'lr': config['train']['lr'],
        'lr_decay_ratio': config['train']['lr_decay_ratio'],
        'patience': config['train']['patience']
    }
    # run, config_wandb = get_wandb_instance(config_wandb, args)
    # test voi 1 tram
    station = target_station[0]
    nan_station = []
    print(station)
    if station != '':
        model = conf['model'](args, config_wandb, nan_station, station, device)
        val_loss= model.train()
        print(val_loss)
        model.test()
        #run.log({"val_loss": val_loss})
