import numpy as np
import scipy
import scipy.stats
import util
import sys, os
import random
from threadpoolctl import threadpool_limits
sys.path.insert(0, os.path.abspath('..'))

from util import Saveable, run_parallel, Singleton

class ICFPMF(Saveable, Singleton):
    
    def __init__(self, num_lat=10, iterations=100, var=0.1, user_var=1.01, item_var=1.01, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.num_lat = num_lat
        self.iterations = iterations
        self.var = var
        self.user_var = user_var
        self.item_var = item_var
        self.user_lambda = self.var/self.user_var
        self.item_lambda = self.var/self.item_var
        self.maps = []
        self.best=None

    def load_var(self, training_matrix):
        self.var = np.var(training_matrix)
        self.user_var = np.mean(np.var(training_matrix,axis=1))
        self.item_var = np.mean(np.var(training_matrix,axis=0))
        self.user_lambda = self.var/self.user_var
        self.item_lambda = self.var/self.item_var

    def fit(self,training_matrix):
        print(self.get_verbose_name())

        self.training_matrix = training_matrix
        num_users = training_matrix.shape[0]
        num_items = training_matrix.shape[1]
        self.lowest_value = lowest_value = np.min(training_matrix)
        highest_value = np.max(training_matrix)
        self.observed_ui = observed_ui = np.nonzero(training_matrix>lowest_value) # itens observed by some user
        # value_abs_range = abs(highest_value - lowest_value)
        self.I = I = np.eye(self.num_lat)

            # self.var = np.mean(np.var(training_matrix))


        self.users_weights = np.random.multivariate_normal(np.zeros(self.num_lat),self.user_var*I,training_matrix.shape[0])
        self.items_weights = np.random.multivariate_normal(np.zeros(self.num_lat),self.item_var*I,training_matrix.shape[1])

        best_map_value = np.inf
        # best_rmse = np.inf
        # self.noise = np.random.normal(self._mean,self.var)
        # self.noise = self._mean
        # #samples
        # without burning
        np.seterr('warn')
        for i in range(self.iterations):
            print(f'[{i+1}/{self.iterations}]')
            # self.noise = np.random.normal(0,self.var)
            # little modified than the original
            # final_users_weights = np.zeros((num_users,self.num_lat))
            # final_items_weights = np.zeros((num_items,self.num_lat))
            with threadpool_limits(limits=1, user_api='blas'):
                for to_run in random.sample([1,2],2):
                    if to_run == 1:
                        self.users_means = np.zeros((num_users,self.num_lat))
                        self.users_covs = np.zeros((num_users,self.num_lat,self.num_lat))
                        args = [(i,) for i in range(num_users)]
                        results = run_parallel(self.compute_user_weight,args)
                        for uid, (mean, cov, weight) in enumerate(results):
                            self.users_means[uid] = mean
                            self.users_covs[uid] = cov
                            self.users_weights[uid] = weight
                    else:
                        self.items_means = np.zeros((num_items,self.num_lat))
                        self.items_covs = np.zeros((num_items,self.num_lat,self.num_lat))
                        args = [(i,) for i in range(num_items)]
                        results = run_parallel(self.compute_item_weight,args)
                        for iid, (mean, cov, weight) in enumerate(results):
                            self.items_means[iid] = mean
                            self.items_covs[iid] = cov
                            self.items_weights[iid] = weight
            
                # final_items_weights[iid] = np.random.multivariate_normal(mean,cov)

            # self.users_weights = final_users_weights
            # self.items_weights = final_items_weights

            rmse=np.sqrt(np.mean((self.get_predicted()[observed_ui] - training_matrix[observed_ui])**2))
            # map_value = 1
            # for val, mean, std in zip(training_matrix[observed_ui],(self.users_weights @ self.items_weights.T)[observed_ui],[self.var]*len(observed_ui[0])):
            #     map_value *= scipy.stats.norm.pdf(val,mean,std)
            map_value = scipy.special.logsumexp(scipy.stats.norm.pdf(training_matrix[observed_ui],(self.users_weights @ self.items_weights.T)[observed_ui],self.var))
                # * scipy.special.logsumexp([scipy.stats.multivariate_normal.pdf(i,np.zeros(self.num_lat),self.user_var*I) for i in self.users_weights.flatten()])\
                # * scipy.special.logsumexp([scipy.stats.multivariate_normal.pdf(i,np.zeros(self.num_lat),self.item_var*I) for i in self.items_weights.flatten()])\
            # map_value = np.prod(r_probabilities)

            print("MAP:",map_value)
            self.maps.append(map_value)
            print("RMSE:",rmse)

            if self.best == None:
                self.best = self.__deepcopy__()
                best_map_value = map_value
            else:
                if map_value > best_map_value:
                    self.best = self.__deepcopy__()
                    best_map_value = map_value

            # print("best =",best_rmse)
            # if self.best == None:
            #     self.best = self.__deepcopy__()
            #     best_rmse = rmse
            # else:
            #     if rmse < best_rmse:
            #         self.best = self.__deepcopy__()
            #         best_rmse = rmse
            # print("best =",best_rmse)
        self = self.best
        del self.best
        del self.training_matrix
        del self.lowest_value
        # del self.noise
        self.save()

    @classmethod
    def compute_user_weight(cls,uid):
        self = cls.getInstance()
        training_matrix = self.training_matrix
        lowest_value = self.lowest_value
        I = self.I
        observed = training_matrix[uid,:]>lowest_value
        tmp = np.linalg.inv((np.dot(self.items_weights[observed].T,self.items_weights[observed]) + I*self.user_lambda))
        mean = tmp.dot(self.items_weights[observed].T).dot(training_matrix[uid,observed])
        cov = tmp*self.var
        return mean, cov, np.random.multivariate_normal(mean,cov)

    @classmethod
    def compute_item_weight(cls,iid):
        self = cls.getInstance()
        training_matrix = self.training_matrix
        lowest_value = self.lowest_value
        I = self.I
        observed = training_matrix[:,iid]>lowest_value
        tmp = np.linalg.inv((np.dot(self.users_weights[observed].T,self.users_weights[observed]) + I*self.item_lambda))
        mean = tmp.dot(self.users_weights[observed].T).dot(training_matrix[observed,iid])
        cov = tmp*self.var
        return mean, cov, np.random.multivariate_normal(mean,cov)

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

    # def get_best_predicted(self):
    #     return self.get_matrix(self.best.users_weights,self.best.items_weights,self.best.var)
