import torch
import os
import argparse
import pandas as pd
import warnings
import logging
from data.dataset import create_dataset, Dataset
from torch.utils.data import DataLoader
from train_eval_sklearn import ml_model_train, ml_model_eval
from train_eval_pytorch import dl_model_train_eval, dl_model_eval
import pickle
from utils.util import load_model

Model = {'ml_model':['LinearRegression', 'DecisionTree', 'RandomForest', 'xgboost'],
         'mlp':['mlp_tabular', 'mlp_sequence'],
         'cnn':['cnn', 'cnn_residual'],
         'rnn': ['rnn', 'lstm', 'gru']
         }


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',level = logging.INFO)
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    parser= argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default = './data', type = str,
                        help='the path to save the dataset')
    parser.add_argument('--dataset_download', action = 'store_true',
                        help = 'download the dataset and split into training, validation and testing set')
    parser.add_argument('--output_root',default = './runs', type = str,
                        help = 'the path to save the model')
    parser.add_argument('--is_feature_eng', action = 'store_true', 
                        help = 'whether use the data after feature engineering or not')
    parser.add_argument('--model_name', default = '', type = str,
                        help = 'select the model')
    parser.add_argument('--do_train_eval', action = 'store_true', 
                        help = 'Training and evaluation')
    parser.add_argument('--do_pred', action = 'store_true',
                        help = 'whether do prediction on test set or not')
    parser.add_argument('--max_depth', type = int, default = 10,
                        help = 'max depth for decision tree')
    parser.add_argument('--min_samples_leaf', type = int, default = 2,
                        help = 'min sample per leaf for decision tree')
    parser.add_argument('--subsample', type = float, default = 0.05,
                        help = 'sample ratio for ensemble models')
    parser.add_argument('--n_estimators', type = int, default = 10,
                        help = 'number of estimators for ensemble models')
    parser.add_argument('--num_epochs', type = int, default = 50,
                        help = 'number of epochs to train the model')
    parser.add_argument('--learning_rate', type = float, default = 5e-4,
                        help = 'learning rate')
    parser.add_argument('--gamma', type = float, default = 0.95,
                        help = 'gamma decay for learning scheduler')
    parser.add_argument('--batch_size', type = int, default = 256,
                        help = 'size of a minibatch')
    parser.add_argument('--weight_decay', type = float, default = 0,
                        help = 'weight decay')
    parser.add_argument('--num_workers', type = int, default = 4,
                        help = 'number of workers')


    args = parser.parse_args()
    
    if args.dataset_download:
        create_dataset(args, logger)


    if args.do_train_eval:
        if args.model_name in Model['ml_model']:
            if args.is_feature_eng:
                logger.info(f' Fitting {args.model_name} model after feature engineering')
            else:
                logger.info(f' Fitting {args.model_name} model without feature engineering')
            model, train_result = ml_model_train(args)
            logger.info(f' Evaluating {args.model_name} model')
            eval_result = ml_model_eval(model, args)
            result_path = os.path.join(args.output_root, args.model_name)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
                pd.DataFrame(train_result).to_csv(os.path.join(result_path, 'train_results.csv'), index = False)
                pd.DataFrame(eval_result).to_csv(os.path.join(result_path, 'eval_results.csv'), index = False)
                logger.info(f' Saving the evaluation result to {os.path.join(result_path, "eval_results.csv")}:')
                print('{\n')
                for key, value in eval_result.items():
                    print(f'    {key}: {value[0]}\n')
                print('}\n')
                with open(os.path.join(result_path, 'model.pkl'), 'wb') as f:
                    logger.info(f' Saving the best model to {os.path.join(result_path, "model.pkl")}')
                    pickle.dump(model, f)
            else:
                best_mse = pd.read_csv(os.path.join(result_path, 'eval_results.csv'))['MSE'].min()
                pd.DataFrame(train_result).to_csv(os.path.join(result_path, 'train_results.csv'), 
                                                                mode = 'a', header = False, index = False)
                pd.DataFrame(eval_result).to_csv(os.path.join(result_path, 'eval_results.csv'), 
                                                                mode = 'a', header= False, index = False)
                logger.info(f' Adding the evaluation result to {os.path.join(result_path, "eval_results.csv")}:')
                print('{\n')
                for key, value in eval_result.items():
                    print(f'    {key}: {value[0]}\n')
                print('}\n')
                if eval_result['MSE'] < best_mse:
                    with open(os.path.join(result_path, 'model.pkl'), 'wb') as f:
                        logger.info(f' Saving the best model to {os.path.join(result_path, "model.pkl")}')
                        pickle.dump(model, f)
        else:
            logger.info(f' Training {args.model_name} model')
            dl_model_train_eval(args, logger)

    if args.do_pred:
        logger.info(f' Evaluating {args.model_name} model on the testing set')
        if args.model_name in Model['ml_model']:
            model = load_model(args, model_type = 'ml')
            test_results = ml_model_eval(model, args)
            print('{\n')
            for key, value in test_results.items():
                print(f'    {key}: {value[0]}\n')
            print('}\n')
        
        else:
            data = DataLoader(Dataset(args, split = 'test'), batch_size = args.batch_size,
                              shuffle = False, num_workers = args.num_workers)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = load_model(args, device)
            model.to(device)
            test_results = dl_model_eval(model, data, device)
            print('{\n')
            for key, value in test_results.items():
                print(f'    {key}: {value[0]}\n')
            print('}\n')