import itertools
from tqdm import tqdm
from util import Saveable
from sklearn.model_selection import KFold, StratifiedKFold
from collections import defaultdict
import scipy.sparse
import numpy as np


class GridSearch(Saveable):
    def __init__(self, model, parameters, n_splits=3):
        self.model = model
        self.parameters = parameters
        self.n_splits = n_splits

    def fit(self, X):
        combinations = list(itertools.product(*list(self.parameters.values())))
        parameters_names = list(self.parameters.keys())
        num_combinations = len(combinations)
        objective_values = defaultdict(list)

        if self.n_splits:
            kf = StratifiedKFold(n_splits=self.n_splits)
            X_data = X.data
            X_row = X.tocoo().row
            X_col = X.tocoo().col

        for i, values in enumerate(combinations):
            print(f'[{i+1}/{num_combinations}]')
            for parameter_name, value in zip(parameters_names, values):
                setattr(self.model, parameter_name, value)
            # print(list(zip(parameters_names,values)))

            if self.n_splits:
                for train_index, test_index in kf.split(X_data, y=X_row):
                    X_train = scipy.sparse.csr_matrix(
                        (X_data[train_index],
                         (X_row[train_index], X_col[train_index])),
                        shape=X.shape)
                    X_test = scipy.sparse.csr_matrix(
                        (X_data[test_index],
                         (X_row[test_index], X_col[test_index])),
                        shape=X.shape)
                    self.model.fit(X_train)
                    score = self.model.score(X_test)
                    objective_values[values].append(score)
                    print('Score:', score)
            else:
                self.model.fit(X)
                objective_values[values].append(self.model.score(X))

        print(objective_values)
        print(
            sorted({k: np.sum(v)
                    for k, v in objective_values.items()}.items(),
                   key=lambda x: x[1],
                   reverse=True))
