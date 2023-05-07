'''machine learning based model'''

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


class ml_model():
    def __init__(self, args):
        if args.model_name == 'LinearRegression':
            self.model = LinearRegression()

        if args.model_name == 'DecisionTree':
            self.model = DecisionTreeRegressor(criterion = 'squared_error', max_depth = args.max_depth,
                                               min_samples_leaf = args.min_samples_leaf, random_state = 9999)
        if args.model_name == 'RandomForest':
            self.model = RandomForestRegressor(criterion = 'squared_error', n_estimators = args.n_estimators,
                                                max_depth = args.max_depth, max_samples = args.subsample,
                                                min_samples_leaf = args.min_samples_leaf, random_state = 9999)
    
        if args.model_name == 'xgboost':
            self.model = XGBRegressor(n_estimators = args.n_estimators, 
                                      min_samples_leaf = args.min_samples_leaf, 
                                      max_depth = args.max_depth, 
                                      subsample = args.subsample,
                                      objective = "reg:squarederror",
                                      seed = 9999)
    
    def get_model(self):
        return self.model