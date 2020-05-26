import itertools
from tqdm import tqdm
from util import Saveable
from sklearn.model_selection import KFold
from collections import defaultdict
import numpy as np
class GridSearch(Saveable):
    def __init__(self, model, parameters, n_splits=5):
        self.model = model
        self.parameters = parameters
        self.n_splits = n_splits

    def fit(self, X):
        kf = KFold(n_splits=self.n_splits)
        combinations = list(itertools.product(*list(self.parameters.values())))
        parameters_names = list(self.parameters.keys())
        num_combinations = len(combinations)
        objective_values = defaultdict(list)

        for i, values in enumerate(combinations):
            print(f'[{i+1}/{num_combinations}]')
            for parameter_name, value in zip(parameters_names,values):
                setattr(self.model,parameter_name,value)
            # print(list(zip(parameters_names,values)))

            for train_index, test_index in kf.split(X.T):
                # print(len(train_index),len(test_index))
                # print("TRAIN:", train_index, "TEST:", test_index)
                # X_train, X_test = X[train_index], X[test_index]
                X_train, X_test = X[:,train_index], X[:,test_index]
                self.model.fit(X_train)
                objective_values[values].append(self.model.score(X_test))
        print(objective_values)
        print(sorted({k: np.sum(v) for k, v in objective_values.items()}.items(), key=lambda x: x[1], reverse=True))
        
        
        
    
