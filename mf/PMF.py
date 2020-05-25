import numpy as np
import scipy
import scipy.stats
import util
import sys, os
import random
from threadpoolctl import threadpool_limits
sys.path.insert(0, os.path.abspath('..'))
import ctypes

from util import Saveable, run_parallel
from mf import MF
class PMF(MF):
    def __init__(self, iterations=100, var=0.1, user_var=1.01, item_var=1.01, learning_rate=1e-3, momentum=0.5, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.iterations = iterations
        self.var = var
        self.user_var = user_var
        self.item_var = item_var
        self.user_lambda = self.var/self.user_var
        self.item_lambda = self.var/self.item_var
        self.objective_values = []
        self.best=None
        self.learning_rate = learning_rate
        self.momentum = momentum

    def load_var(self, training_matrix):
        decimals = 4
        if isinstance(training_matrix,scipy.sparse.spmatrix):
            non_empty_rows = np.sum(training_matrix>0,axis=1).A.flatten()
            training_matrix = training_matrix[non_empty_rows].A
        training_matrix = self.normalize_matrix(training_matrix)
        self.var = np.var(training_matrix)
        # self.user_var = self.var
        # self.item_var = self.var
        self.user_var = np.mean(np.var(training_matrix,axis=1))
        self.item_var = np.mean(np.var(training_matrix,axis=0))
        self.user_lambda = self.var/self.user_var
        self.item_lambda = self.var/self.item_var
        self.var = np.round(self.var,decimals)
        self.user_var = np.round(self.user_var,decimals)
        self.item_var = np.round(self.item_var,decimals)
        self.user_lambda = np.round(self.user_lambda,decimals)
        self.item_lambda = np.round(self.item_lambda,decimals)

    def fit(self,training_matrix):
        super().fit()
        if isinstance(training_matrix,scipy.sparse.spmatrix):
            non_empty_rows = np.sum(training_matrix>0,axis=1).A.flatten()
            training_matrix = training_matrix[non_empty_rows].A
        self_id = id(self)
        training_matrix = self.normalize_matrix(training_matrix)

        num_users = training_matrix.shape[0]
        num_items = training_matrix.shape[1]
        lowest_value = np.min(training_matrix)
        highest_value = np.max(training_matrix)
        indicator_matrix = training_matrix>lowest_value
        observed_ui = np.nonzero(indicator_matrix)
        I = np.eye(self.num_lat)
        self.users_weights = 0.1*np.random.rand(num_users,self.num_lat)
        self.items_weights = 0.1*np.random.rand(num_items,self.num_lat)
        users_momentum = np.zeros(self.users_weights.shape)
        items_momentum = np.zeros(self.items_weights.shape)
        last_objective_value = np.NAN
        last_users_weights = self.users_weights
        last_items_weights = self.items_weights
        np.seterr('warn')
        for i in range(self.iterations):
            print(f'[{i+1}/{self.iterations}]')
            predicted_matrix = self.get_predicted()
            error = indicator_matrix*(training_matrix - util.sigmoid(predicted_matrix))
            users_gradient = error @ (-self.items_weights) + self.user_lambda*self.users_weights
            items_gradient = error.T @ (-self.users_weights) + self.item_lambda*self.items_weights

            users_momentum = self.momentum * users_momentum + self.learning_rate * users_gradient
            items_momentum = self.momentum * items_momentum + self.learning_rate * items_gradient

            self.users_weights -= users_momentum
            self.items_weights -= items_momentum

            objective_value = np.sum((training_matrix[observed_ui] - self.get_predicted()[observed_ui])**2)/2 +\
                self.user_lambda/2 * np.sum(np.linalg.norm(self.users_weights,axis=1)**2) +\
                self.item_lambda/2 * np.sum(np.linalg.norm(self.items_weights,axis=1)**2)
            rmse=np.sqrt(np.mean((self.get_predicted()[observed_ui] - training_matrix[observed_ui])**2))
            print("Objective value",objective_value)
            print("RMSE",rmse)
            if objective_value > last_objective_value:
                print("Achieved convergence with %d iterations, saving %d iteration"%(i+1,i))
                self.users_weights = last_users_weights
                self.items_weights = last_items_weights
                break
            last_objective_value = objective_value
            last_users_weights = self.users_weights.copy()
            last_items_weights = self.items_weights.copy()

    def get_matrix(self, users_weights, items_weights):
        return users_weights @ items_weights.T

    def get_predicted(self):
        return self.get_matrix(self.users_weights,self.items_weights)

    def predict(self,X):
        return self.get_sparse_matrix(self.users_weights,self.items_weights,X)
