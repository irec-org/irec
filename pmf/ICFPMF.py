import numpy as np
import scipy

import sys, os
sys.path.insert(0, os.path.abspath('..'))

from util import Nameable

class ICFPMF(Nameable):
    
    def __init__(self,num_lat=40,iterations=50,var=0.21,us_vars=0.21,is_vars=0.21):
        self.num_lat = num_lat
        self.iterations = iterations
        self.var = var
        self.us_vars = us_vars
        self.is_vars = is_vars
        self.u_lambda = self.var/self.us_vars
        self.i_lambda = self.var/self.is_vars
        self.best=None

    def fit(self,training_matrix, data_var = True):
        num_users = training_matrix.shape[0]
        num_items = training_matrix.shape[1]
        lowest_value = np.min(training_matrix)
        highest_value = np.max(training_matrix)
        self.observed_ui = observed_ui = np.nonzero(training_matrix>lowest_value) # itens observed by some user
        # value_abs_range = abs(highest_value - lowest_value)
        self._mean = np.mean(training_matrix[observed_ui])
        print(f"mean = {self._mean}")
        I = np.eye(self.num_lat)

        if data_var:
            self.us_vars = 1/np.mean(np.var(training_matrix,axis=1))
            self.is_vars = 1/np.mean(np.var(training_matrix,axis=0))
            self.var = np.mean(np.var(training_matrix))

        print(self.us_vars,self.is_vars,self.var)

        self.us_weights = np.random.multivariate_normal(np.zeros(self.num_lat),self.us_vars*I,training_matrix.shape[0])
        self.is_weights = np.random.multivariate_normal(np.zeros(self.num_lat),self.is_vars*I,training_matrix.shape[1])

        # self.noise = np.random.normal(self._mean,self.var)
        # self.noise = self._mean
        # #samples
        # without burning
        for i in range(self.iterations):
            print(f'[{i+1}/{self.iterations}]')
            # self.noise = np.random.normal(self._mean,self.var)
            # self.noise = np.random.normal(sha)
            # little modified than the original
            # final_us_weights = np.zeros((num_users,self.num_lat))
            # final_is_weights = np.zeros((num_items,self.num_lat))

            self.users_means = dict()
            self.users_covs = dict()

            for uid in range(num_users):
                observed = training_matrix[uid,:]>lowest_value
                tmp = np.linalg.inv((np.dot(self.is_weights[observed].T,self.is_weights[observed]) + I*self.u_lambda))
                mean = tmp.dot(self.is_weights[observed].T).dot(training_matrix[uid,observed])
                cov = tmp*self.var
                self.users_means[uid] = mean
                self.users_covs[uid] = cov
                self.us_weights[uid] = np.random.multivariate_normal(mean,cov)
                # final_us_weights[uid] = np.random.multivariate_normal(mean,cov)

            self.items_means = dict()
            self.items_covs = dict()

            for iid in range(num_items):
                observed = training_matrix[:,iid]>lowest_value
                tmp = np.linalg.inv((np.dot(self.us_weights[observed].T,self.us_weights[observed]) + I*self.u_lambda))
                mean = tmp.dot(self.us_weights[observed].T).dot(training_matrix[observed,iid])
                cov = tmp*self.var
                self.items_means[iid] = mean
                self.items_covs[iid] = cov
                self.is_weights[iid] = np.random.multivariate_normal(mean,cov)
                # final_is_weights[iid] = np.random.multivariate_normal(mean,cov)

            # self.us_weights = final_us_weights
            # self.is_weights = final_is_weights

            self.rmse=np.sqrt(np.mean((self.get_predicted()[observed_ui] - training_matrix[observed_ui])**2))

            print("current =",self.rmse)
            if self.best == None:
                self.best = self.__deepcopy__()
            else:
                if self.rmse < self.best.rmse:
                    self.best = self.__deepcopy__()
            print("best =",self.best.rmse)

    def __deepcopy__(self):
        new = type(self)()
        new.__dict__.update(self.__dict__)
        new.us_weights = self.us_weights.copy()
        new.is_weights = self.is_weights.copy()
        return new

    def get_matrix(self, us_weights, is_weights, var):
        return np.random.normal(us_weights @ is_weights.T,var)
    
    def get_predicted(self):
        return self.get_matrix(self.us_weights,self.is_weights,self.var)

    def get_best_predicted(self):
        return self.get_matrix(self.best.us_weights,self.best.is_weights,self.best.var)

    def get_means_and_covs(self):
        return self.best.users_means, self.best.users_covs, self.best.items_means, self.best.items_covs
