from utils.util import get_model
import pandas as pd
import os
from utils.loss import criterion


def ml_model_train(args):
    model = get_model(args, model_type = 'ml')
    if args.is_feature_eng:
        data = pd.read_csv(os.path.join(args.dataset_root, 'trainFE.csv'))
            
    else:
        data = pd.read_csv(os.path.join(args.dataset_root, 'train.csv'))

    X, y = data[data.columns[:-1]], data[data.columns[-1]]
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, criterion(y, y_pred, args)


def ml_model_eval(model, args):
    if args.is_feature_eng:
        if args.do_train_eval:
            data = pd.read_csv(os.path.join(args.dataset_root, 'valFE.csv'))
        if args.do_pred:
            data = pd.read_csv(os.path.join(args.dataset_root, 'testFE.csv'))
    else:
        if args.do_train_eval:
            data = pd.read_csv(os.path.join(args.dataset_root, 'val.csv'))
        if args.do_pred:
            data = pd.read_csv(os.path.join(args.dataset_root, 'test.csv'))
    X, y = data[data.columns[:-1]], data[data.columns[-1]]
    y_pred = model.predict(X)
    return criterion(y, y_pred, args)
