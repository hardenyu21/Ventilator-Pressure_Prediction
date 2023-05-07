import os
import argparse


'''grid search for decision tree, random forest and xgboost'''

if __name__ == '__main__':
    
    parser= argparse.ArgumentParser()
    parser.add_argument('--model_name', default = '',
                        help = 'select the model')
    
    args = parser.parse_args()
    
    if args.model_name == 'DecisionTree':
        for max_depth in range(5, 21):
            command = 'python main.py --do_train_eval' 
            command += f' --model_name DecisionTree'
            command += f' --max_depth {max_depth}'
            command += f' --is_feature_eng'
            os.system(command)
    else:
        for n_estimators in [10, 50, 100]:
            for max_depth in range(5, 16):
                command = 'python main.py --do_train_eval' 
                command += f' --model_name {args.model_name}'
                command += f' --max_depth {max_depth}'
                command += f' --n_estimators {n_estimators}'
                command += f' --is_feature_eng'
                os.system(command)