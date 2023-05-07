import torch
from torch import nn
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE


def MAPE(y_true, y_predict):
    return np.mean(np.abs((y_predict - y_true) / y_true))

def SMAPE(y_true, y_predict):
    return 2.0 * np.mean(np.abs(y_predict - y_true) / (np.abs(y_predict) + np.abs(y_true))) 

def criterion(y_true, y_predict, args):
    r2 = r2_score(y_true, y_predict)
    mse = MSE(y_true, y_predict)
    mae = MAE(y_true, y_predict)
    mape = MAPE(y_true, y_predict)
    smape = SMAPE(y_true, y_predict)
    criterion_results = {'R-Square': [r2], 'MSE': [mse], 'MAE': [mae], 'MAPE': [mape], 'SMAPE': [smape]}
    if args is not None:
        if args.model_name != 'LinearRegression':
            criterion_results['max_depth'] = [args.max_depth]
            criterion_results['min_samples_leaf'] = [args.min_samples_leaf]
            if args.model_name != 'DecisionTree':
                criterion_results['n_estimator'] = [args.n_estimators]
                criterion_results['sample_ratio'] = [args.subsample]
    return criterion_results    

class criterion_calculator():
    def __init__(self):
        self.result = {'Rsquared': [], 'MSE': [], 'MAE': [], 'MAPE': [], 'SMAPE': []}
        self.cache = {'pred': [], 'target': []}
    
    def add_item(self, pred, target):
        self.cache['pred'].append(pred.reshape(-1))
        self.cache['target'].append(target.reshape(-1))
    
    def get_item(self):
        pred = torch.concat(self.cache['pred'])
        target = torch.concat(self.cache['target'])

        SSres = torch.pow(target - pred, 2).sum()
        SStot = torch.pow(target - target.mean(), 2).sum()

        rsquared = 1 - SSres / SStot
        mse = torch.mean(torch.pow(pred - target, 2))
        mae = torch.mean(torch.abs(pred - target))
        mape = torch.mean(torch.abs((pred - target)/ target))
        smape = 2.0 * torch.mean(torch.abs(pred - target) / (torch.abs(pred) + torch.abs(target)))
        self.result['Rsquared'].append(rsquared.cpu().item())
        self.result['MSE'].append(mse.cpu().item())
        self.result['MAE'].append(mae.cpu().item())
        self.result['MAPE'].append(mape.cpu().item())
        self.result['SMAPE'].append(smape.cpu().item())
    
        return self.result

    def initialize(self):
        self.result = {'Rsquared': [], 'MSE': [], 'MAE': [], 'MAPE': [], 'SMAPE': []}
        self.cache = {'pred': [], 'target': []}