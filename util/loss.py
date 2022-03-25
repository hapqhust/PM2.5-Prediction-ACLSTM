import numpy as np 
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import torch

def mape_loss(pred, target, device, reduction='mean'):
    """
    input, output: tensor of same shape
    """
    target = torch.where(
        target == 0, 
        torch.tensor(1e-6), 
        target
    )
    diff = (pred - target) / target
    if reduction == 'mean':
        mape = diff.abs().mean()
    elif reduction == 'sum':
        mape = diff.abs().sum()
    else:
        mape = diff
    return mape