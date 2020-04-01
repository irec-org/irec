import numpy as np
import scipy
from copy import copy

class ICFPMF():
    
    def __init__(self,num_lat,iterations=50,var=0.21,us_vars=0.21,is_vars=0.21):
        self.num_lat = num_lat
        self.iterations = iterations
        self.var = var
        self.us_vars = us_vars
        self.is_vars = is_vars
        self.u_lambda = self.var/self.us_vars
        self.i_lambda = self.var/self.is_vars
        self.best=None

    def fit(self,training_matrix):
        observed_ui = np.nonzero(training_matrix) # itens observed by some user
        num_users = training_matrix.shape[0]
        num_items = training_matrix.shape[1]
        lowest_value = np.min(training_matrix)
        highest_value = np.max(training_matrix)
        # value_abs_range = abs(highest_value - lowest_value)
        self._mean = np.mean(training_matrix[observed_ui])
        print(f"mean = {self._mean}")
        I = np.eye(self.num_lat)

        self.us_weights = np.random.multivariate_normal(np.zeros(self.num_lat),self.us_vars*I,training_matrix.shape[0])
        self.is_weights = np.random.multivariate_normal(np.zeros(self.num_lat),self.is_vars*I,training_matrix.shape[1])

        # #samples
        # without burning
        for i in range(self.iterations):
            print(f'[{i+1}/{self.iterations}]')
            # self.noise = np.random.normal(sha)
            # little modified than the original
            self.noise = np.random.normal(self._mean,self.var)
            final_us_weights = np.zeros((num_users,self.num_lat))
            final_is_weights = np.zeros((num_items,self.num_lat))
            self.users_means = dict()
            self.users_covs = dict()
            for uid in range(num_users):
                observed = training_matrix[uid,:]>lowest_value
                tmp = np.linalg.inv((np.dot(self.is_weights[observed].T,self.is_weights[observed]) + I*self.u_lambda))
                mean = tmp.dot(self.is_weights[observed].T).dot(training_matrix[uid,observed])
                cov = tmp*self.var
                self.users_means[uid] = mean
                self.users_covs[uid] = cov
                final_us_weights[uid] = np.random.multivariate_normal(mean,cov)

            self.items_means = dict()
            self.items_covs = dict()

            for iid in range(num_items):
                observed = training_matrix[:,iid]>lowest_value
                tmp = np.linalg.inv((np.dot(self.us_weights[observed].T,self.us_weights[observed]) + I*self.u_lambda))
                mean = tmp.dot(self.us_weights[observed].T).dot(training_matrix[observed,iid])
                cov = tmp*self.var
                self.items_means[iid] = mean
                self.items_covs[iid] = cov
                final_is_weights[iid] = np.random.multivariate_normal(mean,cov)

            self.us_weights = final_us_weights
            self.is_weights = final_is_weights

            self.rmse=np.sqrt(np.mean((self.get_predicted()[observed_ui] - training_matrix[observed_ui])**2))
            print("current =",self.rmse)
            if self.best == None:
                self.best = copy(self)
            else:
                if self.rmse < self.best.rmse:
                    self.best = copy(self)
            print("best =",self.best.rmse)

    def get_predicted(self):
        return np.dot(self.us_weights,self.is_weights.T) + self.noise

    def get_best_predicted(self):
        return np.dot(self.best.us_weights,self.best.is_weights.T) + self.noise

    def get_means_and_covs(self):
        return self.best.users_means, self.best.users_covs, self.best.items_means, self.best.items_covs

