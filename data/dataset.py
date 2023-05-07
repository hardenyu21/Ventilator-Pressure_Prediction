import torch.utils.data as data
import os
import pandas as pd
import torch
import numpy
from utils.util import new_train_test_split
from utils.feature_eng import Feature_engineering


def create_dataset(args, logger):
    dataset_path = os.path.join(args.dataset_root, 'dataset.csv')
    if not os.path.exists(dataset_path):
        os.system(f'gdown 1CpC725XQEZYyBH_SCdFdGXP7CBGlTlJB --output {dataset_path}')
        logger.info(f' saving the dataset to {dataset_path}')
        logger.info(' Spliting the dataset into training, validation and testing set')
        dataset = pd.read_csv(dataset_path)
        new_train_test_split(dataset, args, is_FE = False)
        logger.info(f' Scale the data by RobustScaler')
        logger.info(f' Saving the training set without feature engineering to {os.path.join(args.dataset_root, "train.csv")}')
        logger.info(f' Saving the validation set without feature engineering to {os.path.join(args.dataset_root, "val.csv")}')
        logger.info(f' Saving the testing set without feature engineeringto {os.path.join(args.dataset_root, "test.csv")}')
        logger.info(' Doing Feature Engineering')
        dataset = Feature_engineering(dataset)
        logger.info(' End Feature Engineering')
        logger.info(' Spliting the dataset after feature enginnering into training, validation and testing set')
        new_train_test_split(dataset, args, is_FE = True)
        logger.info(f' Scale the data by RobustScaler')
        logger.info(f' Saving the training set to {os.path.join(args.dataset_root, "trainFE.csv")}')
        logger.info(f' Saving the validation set to {os.path.join(args.dataset_root, "valFE.csv")}')
        logger.info(f' Saving the testing set to {os.path.join(args.dataset_root, "testFE.csv")}')



'''This is for models by pytorch'''
class Dataset(data.Dataset):
    def __init__(self, args, split):
        target_name = 'pressure'
        self.args = args
        if args.is_feature_eng:
            self.data_path = os.path.join(args.dataset_root, split + 'FE.csv')
        else:
            self.data_path = os.path.join(args.dataset_root, split + '.csv')

        self.data = pd.read_csv(self.data_path)
        columns = list(self.data.columns)
        columns.remove(target_name)

        self.features = torch.tensor(self.data[columns].to_numpy(), dtype = torch.float32)
        self.targets = torch.tensor(self.data[target_name].to_numpy(), dtype = torch.float32)
        self.n_features = self.features.shape[1]
        self.n_records = 80

        if args.model_name != 'mlp_tabular':
            if 'cnn' in args.model_name or 'mlp' in args.model_name: 
                self.features = self.features.reshape(-1, 1, self.n_records, self.n_features)
            else:
                self.features = self.features.reshape(-1, self.n_records, self.n_features)
            self.targets = self.targets.reshape(-1, self.n_records)
        else:
            self.features = self.features.reshape(-1, self.n_features)
            self.targets = self.targets.reshape(-1, 1)

        self.num_samples = self.targets.shape[0]

    def __getitem__(self, idx):
        
        features = self.features
        targets = self.targets
        
        return features[idx], targets[idx]
    
        
    def __len__(self):
        return self.num_samples
    


