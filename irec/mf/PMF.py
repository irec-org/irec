import numpy as np
import scipy
import scipy.stats
import sys, os
import random
from threadpoolctl import threadpool_limits
import ctypes
import metrics
from tqdm import tqdm

from irec.utils.utils import run_parallel
from mf import MF


class PMF(MF):
    def __init__(self,
                 iterations=200,
                 var=0.1,
                 user_var=1,
                 item_var=1,
                 learning_rate=1e-3,
                 momentum=0.6,
                 stop_criteria=0.0009,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.iterations = iterations
        self.var = var
        self.user_var = user_var
        self.item_var = item_var
        self.objective_values = []
        self.best = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.stop_criteria = stop_criteria

    def get_user_lambda(self):
        return self.var / self.user_var

    def get_item_lambda(self):
        return self.var / self.item_var

    def load_var(self, training_matrix):
        decimals = 4
        # if isinstance(training_matrix,scipy.sparse.spmatrix):
        #     non_empty_rows = np.sum(training_matrix>0,axis=1).A.flatten()
        #     training_matrix = training_matrix[non_empty_rows].A
        # training_matrix = self.normalize_matrix(training_matrix)

        # if not isinstance(training_matrix,scipy.sparse.spmatrix):
        #     self.var = np.var(training_matrix)
        #     self.user_var = np.mean(np.var(training_matrix,axis=1))
        #     self.item_var = np.mean(np.var(training_matrix,axis=0))
        # else:
        #     self.var = np.mean(training_matrix.data**2) - np.mean(training_matrix.data)**2
        #     self.user_var = np.mean([np.mean(i.data**2) - np.mean(i.data)**2 if i.getnnz()>0 else 0 for i in training_matrix])
        #     self.item_var = np.mean([np.mean(i.data**2) - np.mean(i.data)**2 if i.getnnz()>0 else 0 for i in training_matrix.transpose()])

        # # self.user_lambda = self.var/self.user_var
        # # self.item_lambda = self.var/self.item_var
        # self.var = np.round(self.var,decimals)
        # self.user_var = np.round(self.user_var,decimals)
        # self.item_var = np.round(self.item_var,decimals)
        # self.user_lambda = np.round(self.user_lambda,decimals)
        # self.item_lambda = np.round(self.item_lambda,decimals)

    def fit(self, training_matrix):
        super().fit()
        # training_matrix = self.normalize_matrix(training_matrix)
        decimals = 4
        user_lambda = self.var / self.user_var
        item_lambda = self.var / self.item_var
        num_users = training_matrix.shape[0]
        num_items = training_matrix.shape[1]
        lowest_value = np.min(training_matrix)
        highest_value = np.max(training_matrix)
        if not isinstance(training_matrix, scipy.sparse.spmatrix):
            indicator_matrix = training_matrix > lowest_value
            observed_ui = np.nonzero(indicator_matrix)
        else:
            observed_ui = (training_matrix.tocoo().row,
                           training_matrix.tocoo().col)
            # observed_ui = list(zip(*observed_ui))
        I = np.eye(self.num_lat)
        self.users_weights = 0.1 * np.random.rand(num_users, self.num_lat)
        self.items_weights = 0.1 * np.random.rand(num_items, self.num_lat)
        users_momentum = np.zeros(self.users_weights.shape)
        items_momentum = np.zeros(self.items_weights.shape)
        last_objective_value = np.NAN
        last_users_weights = self.users_weights
        last_items_weights = self.items_weights
        np.seterr('warn')
        predicted = self.predict(observed_ui)

        tq = tqdm(range(self.iterations))
        for i in tq:
            # print(f'[{i+1}/{self.iterations}]')
            error = scipy.sparse.csr_matrix(
                (training_matrix.data - predicted, observed_ui),
                shape=training_matrix.shape)
            users_gradient = error @ (
                -self.items_weights) + user_lambda * self.users_weights
            items_gradient = error.T @ (
                -self.users_weights) + item_lambda * self.items_weights

            users_momentum = self.momentum * users_momentum + self.learning_rate * users_gradient
            items_momentum = self.momentum * items_momentum + self.learning_rate * items_gradient

            self.users_weights -= users_momentum
            self.items_weights -= items_momentum

            predicted = self.predict(observed_ui)

            # objective_value = np.sum((training_matrix.data - predicted)**2)/2 +\
            #     user_lambda/2 * np.sum(np.linalg.norm(self.users_weights,axis=1)**2) +\
            #     item_lambda/2 * np.sum(np.linalg.norm(self.items_weights,axis=1)**2)
            # print("Objective value",objective_value)

            rmse = metrics.rmse(training_matrix.data, predicted)
            objective_value = rmse
            # print("RMSE",rmse)
            if objective_value > last_objective_value or np.fabs(
                    objective_value -
                    last_objective_value) <= self.stop_criteria:
                print(
                    "Achieved convergence with %d iterations, saving %d iteration"
                    % (i + 1, i))
                self.users_weights = last_users_weights
                self.items_weights = last_items_weights
                break

            last_objective_value = objective_value
            self.objective_value = objective_value
            last_users_weights = self.users_weights.copy()
            last_items_weights = self.items_weights.copy()

            tq.set_description('cur={:.3f},last={:.3f}'.format(
                objective_value, last_objective_value))
