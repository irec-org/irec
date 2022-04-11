import numpy as np
import scipy.stats
import random
from threadpoolctl import threadpool_limits
import ctypes

from irec.utils.utils import run_parallel
from .MF import MF


class ICFPMF(MF):
    def __init__(self, iterations=50, var=1, user_var=1, item_var=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterations = iterations
        self.var = var
        self.user_var = user_var
        self.item_var = item_var
        self.user_lambda = self.var / self.user_var
        self.item_lambda = self.var / self.item_var
        self.objective_values = []
        self.best = None

    def load_var(self, training_matrix):
        decimals = 4
        training_matrix = self.normalize_matrix(training_matrix)
        self.var = np.var(training_matrix)
        # self.user_var = self.var
        # self.item_var = self.var
        self.user_var = np.mean(np.var(training_matrix, axis=1))
        self.item_var = np.mean(np.var(training_matrix, axis=0))
        self.user_lambda = self.var / self.user_var
        self.item_lambda = self.var / self.item_var
        self.var = np.round(self.var, decimals)
        self.user_var = np.round(self.user_var, decimals)
        self.item_var = np.round(self.item_var, decimals)
        self.user_lambda = np.round(self.user_lambda, decimals)
        self.item_lambda = np.round(self.item_lambda, decimals)

    def fit(self, training_matrix):
        super().fit()
        training_matrix = self.normalize_matrix(training_matrix)
        self_id = id(self)
        self.training_matrix = training_matrix
        num_users = training_matrix.shape[0]
        num_items = training_matrix.shape[1]
        self.lowest_value = lowest_value = np.min(training_matrix)
        highest_value = np.max(training_matrix)
        self.observed_ui = observed_ui = np.nonzero(
            training_matrix > lowest_value
        )  # itens observed by some user
        self.I = I = np.eye(self.num_lat)
        self.users_weights = np.random.multivariate_normal(
            np.zeros(self.num_lat), self.user_var * I, training_matrix.shape[0]
        )
        self.items_weights = np.random.multivariate_normal(
            np.zeros(self.num_lat), self.item_var * I, training_matrix.shape[1]
        )
        best_objective_value = np.inf
        # without burning
        np.seterr("warn")
        for i in range(self.iterations):
            print(f"[{i+1}/{self.iterations}]")
            with threadpool_limits(limits=1, user_api="blas"):
                for to_run in random.sample([1, 2], 2):
                    if to_run == 1:
                        self.users_means = np.zeros((num_users, self.num_lat))
                        self.users_covs = np.zeros(
                            (num_users, self.num_lat, self.num_lat)
                        )
                        args = [
                            (
                                self_id,
                                i,
                            )
                            for i in range(num_users)
                        ]
                        results = run_parallel(self.compute_user_weight, args)
                        for uid, (mean, cov, weight) in enumerate(results):
                            self.users_means[uid] = mean
                            self.users_covs[uid] = cov
                            self.users_weights[uid] = weight
                    else:
                        self.items_means = np.zeros((num_items, self.num_lat))
                        self.items_covs = np.zeros(
                            (num_items, self.num_lat, self.num_lat)
                        )
                        args = [
                            (
                                self_id,
                                i,
                            )
                            for i in range(num_items)
                        ]
                        results = run_parallel(self.compute_item_weight, args)
                        for iid, (mean, cov, weight) in enumerate(results):
                            self.items_means[iid] = mean
                            self.items_covs[iid] = cov
                            self.items_weights[iid] = weight

            rmse = np.sqrt(
                np.mean((self.predict(observed_ui) - training_matrix[observed_ui]) ** 2)
            )
            # map_value = scipy.special.logsumexp(scipy.stats.norm.pdf(training_matrix[observed_ui],(self.users_weights @ self.items_weights.T)[observed_ui],self.var))\
            #         * scipy.special.logsumexp([scipy.stats.multivariate_normal.pdf(weights,means,covs)
            #                                for means, covs, weights in zip(self.users_means,self.users_covs,self.users_weights)])\
            #         * scipy.special.logsumexp([scipy.stats.multivariate_normal.pdf(weights,means,covs)
            #                                for means, covs, weights in zip(self.items_means,self.items_covs,self.items_weights)])

            objective_value = (
                np.sum((training_matrix[observed_ui] - self.predict(observed_ui)) ** 2)
                / 2
                + self.user_lambda
                / 2
                * np.sum(np.linalg.norm(self.users_weights, axis=1) ** 2)
                + self.item_lambda
                / 2
                * np.sum(np.linalg.norm(self.items_weights, axis=1) ** 2)
            )

            print("Objective value", objective_value)
            #     self.objective_values.append(objective_value)
            print("RMSE", rmse)

            # if self.best == None:
            #     self.best = self.__deepcopy__()
            #     best_objective_value = objective_value
            # else:
            #     if objective_value < best_objective_value:
            #         self.best = self.__deepcopy__()
            #         best_objective_value = objective_value

        # self = self.best
        # del self.best
        del self.training_matrix
        del self.lowest_value

    @staticmethod
    def compute_user_weight(obj_id, uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        training_matrix = self.training_matrix
        lowest_value = self.lowest_value
        I = self.I
        observed = training_matrix[uid, :] > lowest_value
        tmp = np.linalg.inv(
            (
                np.dot(self.items_weights[observed].T, self.items_weights[observed])
                + I * self.user_lambda
            )
        )
        mean = tmp.dot(self.items_weights[observed].T).dot(
            training_matrix[uid, observed]
        )
        cov = tmp * self.var
        return mean, cov, np.random.multivariate_normal(mean, cov)

    @staticmethod
    def compute_item_weight(obj_id, iid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        training_matrix = self.training_matrix
        lowest_value = self.lowest_value
        I = self.I
        observed = training_matrix[:, iid] > lowest_value
        tmp = np.linalg.inv(
            (
                np.dot(self.users_weights[observed].T, self.users_weights[observed])
                + I * self.item_lambda
            )
        )
        mean = tmp.dot(self.users_weights[observed].T).dot(
            training_matrix[observed, iid]
        )
        cov = tmp * self.var
        return mean, cov, np.random.multivariate_normal(mean, cov)

    def __deepcopy__(self):
        new = type(self)()
        new.__dict__.update(self.__dict__)
        new.users_weights = self.users_weights.copy()
        new.users_means = self.users_means.copy()
        new.users_covs = self.users_covs.copy()
        new.items_weights = self.items_weights.copy()
        new.items_means = self.items_means.copy()
        new.items_covs = self.items_covs.copy()
        return new
