import numpy as np
import scipy

import sys, os
sys.path.insert(0, os.path.abspath('..'))

from util import Saveable

class ICFPMF(Saveable):
    
    def __init__(self, num_lat=40, iterations=5, var=0.21, user_var=0.21, item_var=0.21, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.num_lat = num_lat
        self.iterations = iterations
        self.var = var
        self.user_var = user_var
        self.item_var = item_var
        self.user_lambda = self.var/self.user_var
        self.item_lambda = self.var/self.item_var
        self.best=None

    def load_var(self, training_matrix):
        self.user_var = 1/np.mean(np.var(training_matrix,axis=1))
        self.item_var = 1/np.mean(np.var(training_matrix,axis=0))
        self.user_lambda = self.var/self.user_var
        self.item_lambda = self.var/self.item_var

    def fit(self,training_matrix):
        print(self.get_verbose_name())
        num_users = training_matrix.shape[0]
        num_items = training_matrix.shape[1]
        lowest_value = np.min(training_matrix)
        highest_value = np.max(training_matrix)
        self.observed_ui = observed_ui = np.nonzero(training_matrix>lowest_value) # itens observed by some user
        # value_abs_range = abs(highest_value - lowest_value)
        I = np.eye(self.num_lat)

            # self.var = np.mean(np.var(training_matrix))


        self.users_weights = np.random.multivariate_normal(np.zeros(self.num_lat),self.user_var*I,training_matrix.shape[0])
        self.items_weights = np.random.multivariate_normal(np.zeros(self.num_lat),self.item_var*I,training_matrix.shape[1])

        best_rmse = np.inf
        # self.noise = np.random.normal(self._mean,self.var)
        # self.noise = self._mean
        # #samples
        # without burning
        for i in range(self.iterations):
            print(f'[{i+1}/{self.iterations}]')
            # self.noise = np.random.normal(self._mean,self.var)
            # self.noise = np.random.normal(sha)
            # little modified than the original
            # final_users_weights = np.zeros((num_users,self.num_lat))
            # final_items_weights = np.zeros((num_items,self.num_lat))

            self.users_means = dict()
            self.users_covs = dict()

            for uid in range(num_users):
                observed = training_matrix[uid,:]>lowest_value
                tmp = np.linalg.inv((np.dot(self.items_weights[observed].T,self.items_weights[observed]) + I*self.user_lambda))
                mean = tmp.dot(self.items_weights[observed].T).dot(training_matrix[uid,observed])
                cov = tmp*self.var
                self.users_means[uid] = mean
                self.users_covs[uid] = cov
                self.users_weights[uid] = np.random.multivariate_normal(mean,cov)
                # final_users_weights[uid] = np.random.multivariate_normal(mean,cov)

            self.items_means = dict()
            self.items_covs = dict()

            for iid in range(num_items):
                observed = training_matrix[:,iid]>lowest_value
                tmp = np.linalg.inv((np.dot(self.users_weights[observed].T,self.users_weights[observed]) + I*self.item_lambda))
                mean = tmp.dot(self.users_weights[observed].T).dot(training_matrix[observed,iid])
                cov = tmp*self.var
                self.items_means[iid] = mean
                self.items_covs[iid] = cov
                self.items_weights[iid] = np.random.multivariate_normal(mean,cov)
                # final_items_weights[iid] = np.random.multivariate_normal(mean,cov)

            # self.users_weights = final_users_weights
            # self.items_weights = final_items_weights

            rmse=np.sqrt(np.mean((self.get_predicted()[observed_ui] - training_matrix[observed_ui])**2))

            print("current =",rmse)
            if self.best == None:
                self.best = self.__deepcopy__()
                best_rmse = rmse
            else:
                if rmse < best_rmse:
                    self.best = self.__deepcopy__()
                    best_rmse = rmse
            print("best =",best_rmse)
        self = self.best
        del self.best
        self.save()
    def __deepcopy__(self):
        new = type(self)()
        new.__dict__.update(self.__dict__)
        new.users_weights = self.users_weights.copy()
        new.items_weights = self.items_weights.copy()
        return new

    def get_matrix(self, users_weights, items_weights, var):
        return np.random.normal(users_weights @ items_weights.T,var)
    
    def get_predicted(self):
        return self.get_matrix(self.users_weights,self.items_weights,self.var)

    def get_best_predicted(self):
        return self.get_matrix(self.best.users_weights,self.best.items_weights,self.best.var)
