import numpy as np
import scipy
import scipy.stats
from threadpoolctl import threadpool_limits
import ctypes
import collections
import metrics
from tqdm import tqdm

from irec.utils.utils import run_parallel
from .MF import MF
from numba import jit


@jit(nopython=True)
def _multivariate_normal(x, mu=0, sigma=1):
    """focus in speed and readability multivariate_normal, no checks like scipy and others libraries
be sure to pass float array and matrix
very limited to be more fast"""
    k = len(x)
    return (1/(np.sqrt(2*np.pi)**k * np.linalg.det(sigma)))*\
        np.exp((-1/2)*(x-mu) @ np.linalg.inv(sigma)@(x - mu))


@jit(nopython=True)
def _apply_multivariate_normal(xs, mu=0, sigma=1):
    n = len(xs)
    result = np.zeros(n)
    for i in range(n):
        result[i] = _multivariate_normal(xs[i], mu, sigma)
    return result


def _norm_sum_probabilities(x):
    return np.sum(-np.log(x))
    # return scipy.special.logsumexp(x)


def _norm_ratings(x, highest_value, lowest_value):
    return 2 * (x - lowest_value) / (highest_value - lowest_value) - 1


def _unnorm_ratings(x, highest_value, lowest_value):
    return (x + 1) / 2 * (highest_value - lowest_value) + lowest_value


class ICFPMFS(MF):
    def __init__(self,
                 iterations=20,
                 var=0.05,
                 user_var=0.01,
                 item_var=0.01,
                 stop_criteria=0.0009,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.iterations = iterations
        self.var = var
        self.user_var = user_var
        self.item_var = item_var
        self.stop_criteria = stop_criteria

    def get_user_lambda(self):
        return self.var / self.user_var

    def get_item_lambda(self):
        return self.var / self.item_var

    def load_var(self, training_matrix):
        decimals = 4
        # self.var = np.mean(training_matrix.data**2) - np.mean(training_matrix.data)**2
        # self.user_var = np.mean([np.mean(i.data**2) - np.mean(i.data)**2 if i.getnnz()>0 else 0 for i in training_matrix])
        # self.item_var = np.mean([np.mean(i.data**2) - np.mean(i.data)**2 if i.getnnz()>0 else 0 for i in training_matrix.transpose()])

        # self.var = np.round(self.var,decimals)
        # self.user_var = np.round(self.user_var,decimals)
        # self.item_var = np.round(self.item_var,decimals)

    def fit(self, training_matrix):
        super().fit()
        train_uids = np.unique(training_matrix.tocoo().row)
        self.objective_values = []
        self.best = None
        decimals = 4
        self.user_lambda = self.var / self.user_var
        self.item_lambda = self.var / self.item_var
        self.r_mean = np.mean(training_matrix.data)
        # self.r_mean = np.
        self_id = id(self)
        # training_matrix = self.normalize_matrix(training_matrix)

        self.training_matrix = training_matrix
        num_users = training_matrix.shape[0]
        num_items = training_matrix.shape[1]
        self.lowest_value = lowest_value = np.min(training_matrix)
        self.highest_value = highest_value = np.max(training_matrix)
        self.observed_ui = observed_ui = (training_matrix.tocoo().row,
                                          training_matrix.tocoo().col
                                          )  # itens observed by some user
        self.I = I = np.eye(self.num_lat)
        self.users_weights = np.random.multivariate_normal(
            np.zeros(self.num_lat), self.user_var * I,
            training_matrix.shape[0])
        self.items_weights = np.random.multivariate_normal(
            np.zeros(self.num_lat), self.item_var * I,
            training_matrix.shape[1])
        # self.users_weights[~np.isin(list(range(self.users_weights.shape[0])), train_uids)] = np.ones(self.num_lat)

        # self.users_weights = np.mean(training_matrix.data)*np.random.rand(num_users,self.num_lat)
        # self.items_weights = np.mean(training_matrix.data)*np.random.rand(num_items,self.num_lat)
        # self.users_weights = 0.1*np.random.multivariate_normal(np.zeros(self.num_lat),self.user_var*I,training_matrix.shape[0])
        # self.items_weights = 0.1*np.random.multivariate_normal(np.zeros(self.num_lat),self.item_var*I,training_matrix.shape[1])

        self.users_observed_items = collections.defaultdict(list)
        self.items_observed_users = collections.defaultdict(list)
        self.users_observed_items_ratings = collections.defaultdict(list)
        self.items_observed_users_ratings = collections.defaultdict(list)
        for uid, iid in zip(*self.observed_ui):
            self.users_observed_items[uid].append(iid)
            self.items_observed_users[iid].append(uid)

        for uid, iids in self.users_observed_items.items():
            self.users_observed_items_ratings[uid] = training_matrix[uid,
                                                                     iids].data

        for iid, uids in self.items_observed_users.items():
            self.items_observed_users_ratings[iid] = training_matrix[uids,
                                                                     iid].data

        best_objective_value = None

        last_objective_value = None

        # users_probabilities_args = [(self_id, i) for i in range(self.users_weights.shape[0])]
        # items_probabilities_args = [(self_id, i) for i in range(self.items_weights.shape[0])]

        # without burning
        np.seterr('warn')
        tq = tqdm(range(self.iterations))
        for i in tq:
            with threadpool_limits(limits=1, user_api='blas'):
                # for to_run in random.sample([1,2],2):
                for to_run in [1, 2]:
                    if to_run == 1:
                        self.users_means = np.zeros((num_users, self.num_lat))
                        self.users_covs = np.zeros(
                            (num_users, self.num_lat, self.num_lat))
                        args = [(
                            self_id,
                            i,
                        ) for i in train_uids]
                        results = run_parallel(self.compute_user_weight,
                                               args,
                                               use_tqdm=False)
                        for uid, (mean, cov,
                                  weight) in zip(train_uids, results):
                            self.users_means[uid] = mean
                            self.users_covs[uid] = cov
                            self.users_weights[uid] = weight
                    else:
                        self.items_means = np.zeros((num_items, self.num_lat))
                        self.items_covs = np.zeros(
                            (num_items, self.num_lat, self.num_lat))
                        args = [(
                            self_id,
                            i,
                        ) for i in range(num_items)]
                        results = run_parallel(self.compute_item_weight,
                                               args,
                                               use_tqdm=False)
                        for iid, (mean, cov, weight) in enumerate(results):
                            self.items_means[iid] = mean
                            self.items_covs[iid] = cov
                            self.items_weights[iid] = weight

            predicted = self.predict(observed_ui)

            tq.set_description('rmse={:.3f}'.format(
                metrics.rmse(
                    training_matrix.data,
                    _unnorm_ratings(predicted, self.lowest_value,
                                    self.highest_value))))
            tq.refresh()

        #     # objective_value = _norm_sum_probabilities(scipy.stats.norm.pdf(training_matrix.data,predicted,self.var))\
        #     #     + _norm_sum_probabilities(_apply_multivariate_normal(self.users_weights,np.zeros(self.num_lat),self.var*self.I))\
        #     #     + _norm_sum_probabilities(_apply_multivariate_normal(self.items_weights,np.zeros(self.num_lat),self.var*self.I))

        #     objective_value = np.sum((training_matrix.data - predicted)**2)/2 +\
        #         self.user_lambda/2 * np.sum(np.linalg.norm(self.users_weights,axis=1)**2) +\
        #         self.item_lambda/2 * np.sum(np.linalg.norm(self.items_weights,axis=1)**2)

        #     self.objective_values.append(objective_value)

        #     if self.best == None:
        #         self.best = self.__deepcopy__()
        #         best_objective_value = objective_value
        #     else:
        #         if objective_value < best_objective_value:
        #             self.best = self.__deepcopy__()
        #             best_objective_value = objective_value

        #     tq.set_description('cur={:.3f},best={:.3f}'.format(objective_value,best_objective_value))
        #     tq.refresh()

        #     # predicted = self.predict(observed_ui)
        #     # rmse=metrics.rmse(training_matrix.data,predicted)
        #     # objective_value = rmse
        #     # print("RMSE",rmse)
        #     # if np.fabs(objective_value - last_objective_value) <= self.stop_criteria:
        #     #     self.objective_value = objective_value
        #     #     print("Achieved convergence with %d iterations"%(i+1))
        #     #     break
        #     # last_objective_value = objective_value

        #     # sparse_predicted = self.get_sparse_predicted(observed_ui_pair)
        #     # rmse=np.sqrt(np.mean((sparse_predicted - training_matrix.data)**2))
        #     # objective_value = np.sum((training_matrix.data - sparse_predicted)**2)/2 +\
        #     #     self.user_lambda/2 * np.sum(np.linalg.norm(self.users_weights,axis=1)**2) +\
        #     #     self.item_lambda/2 * np.sum(np.linalg.norm(self.items_weights,axis=1)**2)
        #     # print("Objective value",objective_value)
        #     # #     self.objective_values.append(objective_value)
        #     # print("RMSE",rmse)
        # self.__dict__.update(self.best.__dict__)
        # del self.best
        del self.r_mean
        del self.user_lambda
        del self.item_lambda
        del self.users_observed_items
        del self.users_observed_items_ratings
        del self.items_observed_users
        del self.items_observed_users_ratings
        del self.observed_ui
        del self.I
        del self.training_matrix
        del self.lowest_value
        del self.highest_value

    @staticmethod
    def _user_probability(obj_id, uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        return scipy.stats.multivariate_normal.pdf(self.users_weights[uid],
                                                   np.zeros(self.num_lat),
                                                   self.var * self.I)

    @staticmethod
    def _item_probability(obj_id, iid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        return scipy.stats.multivariate_normal.pdf(self.items_weights[iid],
                                                   np.zeros(self.num_lat),
                                                   self.var * self.I)

    @staticmethod
    def compute_user_weight(obj_id, uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        training_matrix = self.training_matrix
        lowest_value = self.lowest_value
        I = self.I
        observed = self.users_observed_items[uid]
        tmp = np.linalg.inv(
            (np.dot(self.items_weights[observed].T,
                    self.items_weights[observed]) + I * self.user_lambda))
        mean = tmp.dot(self.items_weights[observed].T).dot(
            _norm_ratings(self.users_observed_items_ratings[uid],
                          self.lowest_value, self.highest_value))
        cov = tmp * self.var
        return mean, cov, np.random.multivariate_normal(mean, cov)
        # return mean, cov, scipy.stats.multivariate_normal.pdf(self.users_weights[uid],mean,cov)
        # return mean, cov, scipy.stats.multivariate_normal.pdf(np.random.multivariate_normal(mean,cov),mean,cov)

    @staticmethod
    def compute_item_weight(obj_id, iid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        training_matrix = self.training_matrix
        lowest_value = self.lowest_value
        I = self.I
        observed = self.items_observed_users[iid]
        tmp = np.linalg.inv(
            (np.dot(self.users_weights[observed].T,
                    self.users_weights[observed]) + I * self.item_lambda))
        mean = tmp.dot(self.users_weights[observed].T).dot(
            _norm_ratings(self.items_observed_users_ratings[iid],
                          self.lowest_value, self.highest_value))
        cov = tmp * self.var
        return mean, cov, np.random.multivariate_normal(mean, cov)
        # return mean, cov, scipy.stats.multivariate_normal.pdf(self.items_weights[iid],mean,cov)
        # return mean, cov, scipy.stats.multivariate_normal.pdf(np.random.multivariate_normal(mean,cov),mean,cov)

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
