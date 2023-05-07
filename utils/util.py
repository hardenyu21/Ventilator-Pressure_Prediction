from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import pandas as pd
import os
from models.ml import ml_model
from models.dl import get_mlp, get_cnn, get_rnn
import pickle
import torch

def new_train_test_split(dataset, args, is_FE):
    n = 80                              ##number of records for each unique breath_id
    RS = RobustScaler()
    columns = list(dataset.columns)
    columns.remove('pressure')
    columns.remove('id')
    columns.remove('breath_id')
    target = dataset['pressure'].to_numpy().reshape(-1, n, 1)
    features = dataset[columns].to_numpy()
    n_features = features.shape[1]
    features = RS.fit_transform(features).reshape(-1, n, n_features)

    '''split ratio: 2 : 1 : 1'''
    X_train, X_val_test, y_train, y_val_test = train_test_split(features, target, test_size = 0.5, random_state = 9999)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size = 0.5, random_state = 9999)
    X_train, X_val, X_test = X_train.reshape(-1, n_features), X_val.reshape(-1, n_features), X_test.reshape(-1, n_features)
    y_train, y_val, y_test = y_train.reshape(-1, 1), y_val.reshape(-1, 1), y_test.reshape(-1, 1)

    training_set = pd.concat([pd.DataFrame(X_train, columns = columns), 
                              pd.DataFrame(y_train, columns = ['pressure'])], axis = 1)
    validation_set = pd.concat([pd.DataFrame(X_val, columns = columns), 
                                pd.DataFrame(y_val, columns = ['pressure'])], axis = 1)
    testing_set = pd.concat([pd.DataFrame(X_test, columns = columns), 
                             pd.DataFrame(y_test, columns = ['pressure'])], axis = 1)
    
    if is_FE:
        training_set.to_csv(os.path.join(args.dataset_root, 'trainFE.csv'), index = False)
        validation_set.to_csv(os.path.join(args.dataset_root, 'valFE.csv'), index = False)
        testing_set.to_csv(os.path.join(args.dataset_root, 'testFE.csv'), index = False)
    else:
        training_set.to_csv(os.path.join(args.dataset_root, 'train.csv'), index = False)
        validation_set.to_csv(os.path.join(args.dataset_root, 'val.csv'), index = False)
        testing_set.to_csv(os.path.join(args.dataset_root, 'test.csv'), index = False)


def get_model(args, model_type = None):

    if model_type == 'ml':
        return ml_model(args).get_model()

    else:
        if 'mlp' in args.model_name:
            return get_mlp(args)
        
        if 'cnn' in args.model_name:
            return get_cnn(args)
        
        else:
            return get_rnn(args)

def load_model(args, device, model_type = None):

    if model_type == 'ml':
        with open(os.path.join(args.output_root, args.model_name, 'model.pkl'), 'rb') as f:
            model = pickle.load(f)

    else:
        model = get_model(args)
        model.load_state_dict(torch.load(os.path.join(args.output_root, args.model_name, 'model.pt'),
                                         map_location = device))
    
    return model
