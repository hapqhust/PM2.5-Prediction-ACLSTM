import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_lstm_att.model import CNNLSTMAttention
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.early_stop import EarlyStopping
from models.cnn_lstm_att.utils import get_dataloader, preprocess, Generator
from util import model as model_utils
import numpy as np

class CNN_LSTM_ATTSupervisor():
    def __init__(self, args, config, nan_stations, station_name, device):
        # Config
        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.batch_size = config['batch_size']
        self.device = device
        # self._index = station_idx
        self._name = station_name
        self.config = config

        # Data
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.scale_y = preprocess(config, nan_stations, station_name)
        self._base_dir = model_utils.generate_log_dir(args)
        self._weights_path = os.path.join(self._base_dir, "best.pth")

        # Model
        self.model = CNNLSTMAttention(config, device).to(self.device)
        self._es = EarlyStopping(
            patience=self.config['patience'],
            verbose=True,
            delta=0.0,
            path=self._weights_path            
        )


    def train(self):

        train_dataset = Generator(self.X_train, self.y_train)
        train_dataloader = get_dataloader(train_dataset, self.batch_size, shuffle = False)
        valid_dataset = Generator(self.X_valid, self.y_valid)
        valid_dataloader = get_dataloader(valid_dataset, self.batch_size, shuffle = False)
        
        model = self.model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.config['lr_decay_ratio'], patience = 2)

        
        for epoch in range(self.epochs):
            if not self._es.early_stop:
                
                #Train
                train_losses = []
                epoch_train_loss = 0
                model.train()
                for (idx, (input, target)) in enumerate(train_dataloader):
                    input, target = input.float().to(self.device), target.float().to(self.device)
                    optimizer.zero_grad()
                    out = model(input)
                    loss = criterion(out, target)
                    epoch_train_loss += loss.item()
                    loss.backward() 
                    optimizer.step()
                train_losses.append(epoch_train_loss/len(train_dataloader))
                epoch_train_loss = 0;

                #Validation
                epoch_val_loss = 0
                val_losses = []
                model.eval()
                with torch.no_grad():
                    for (idx, (input, target)) in enumerate(valid_dataloader):
                        input, target = input.float().to(self.device), target.float().to(self.device)
                        out = model(input)
                        loss = criterion(out, target)
                        epoch_val_loss += loss.item()
                    val_loss = epoch_val_loss / len(valid_dataloader)

                val_losses.append(val_loss)
                scheduler.step(val_loss)
                print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
                self._es(val_loss, model)
        return val_losses[-1]

    #Test model
    def test(self):
        self.model.load_state_dict(torch.load(self._weights_path)["model_dict"])
        self.model.eval()
        
        test_dataset = Generator(self.X_test, self.y_test)
        test_dataloader = get_dataloader(test_dataset, self.batch_size, shuffle = False)

        list_predict = []
        list_actual = []
        with torch.no_grad():
            for(idx, (input, target)) in enumerate(test_dataloader):
                input, target = input.float().to(self.device), target.float().to(self.device)
                out = self.model(input)
                list_predict.append(out.cpu().detach().numpy())
                list_actual.append(target.cpu().detach().numpy())
                #loss = criterion(out, target)
                #print('loss:', loss.item())

        final_out = np.concatenate(list_predict, axis = 0)
        final_target = np.concatenate(list_actual, axis = 0)

        out = self.scale_y.inverse_transform(final_out.reshape(-1,1))
        target = self.scale_y.inverse_transform(final_target.reshape(-1,1))

        print("Value of MAE = {}".format(model_utils.mae(target, out)))
        print("Value of RMSE = {}".format(model_utils.rmse(target, out)))
        print("Value of MAPE = {}".format(model_utils.mape(target, out)))
        # model_utils.visualize(target, out, self._base_dir, self._name)
        model_utils.save_results(target, out, self._base_dir, self._name)
        model_utils.visualize(target, out, self._base_dir, "result_{}_{}h.png".format(self._name, self.config['horizon']) )

