from .ICF import ICF
from .LinearICF import LinearICF
import numpy as np
from tqdm import tqdm
#import util
from threadpoolctl import threadpool_limits
import scipy.optimize
import ctypes
from collections import defaultdict
import joblib
import scipy
import mf
from lib.utils.PersistentDataManager import PersistentDataManager
import interactors

class GLM_UCB(LinearICF):
    def __init__(self, c=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.parameters.extend(['c'])

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def p(self,x):
        return x

    def train(self,train_dataset):
        super().train(train_dataset)
        self.users_rec_rewards = defaultdict(list)
        self.users_rec_items_means = defaultdict(list)
        self.p_vals = dict()

    def error_user_weight_function(self,p,u_rec_rewards,u_rec_items_means):
        return np.sum(np.array(
            [(u_rec_rewards[t] - self.p(p.T @ u_rec_items_means[t]))*u_rec_items_means[t]
             for t in range(0,len(u_rec_items_means))]),0)

    def predict(self,uid,candidate_items,num_req_items):
        A = self.As[uid]
        if len(self.users_rec_items_means[uid]) == 0:
            self.p_vals[uid] = np.zeros(self.num_latent_factors)
        else:
            self.p_vals[uid] = scipy.optimize.root(self.error_user_weight_function,
                                    self.p_vals[uid],
                                    (self.users_rec_rewards[uid],self.users_rec_items_means[uid])).x
        cov = np.linalg.inv(A)*self.var
        items_score = self.p(self.p_vals[uid][None,:] @ self.items_means[candidate_items].T) +\
            self.c * np.sqrt(np.log(self.t+1)) *\
            np.sqrt(np.sum(self.items_means[candidate_items].dot(cov) *\
                           self.items_means[candidate_items],axis=1))
        items_score = items_score.flatten()
        return items_score, None
    def update(self,uid,item,reward,additional_data):
        max_item_mean = self.items_means[item,:]
        self.users_rec_rewards[uid].append(reward)
        self.users_rec_items_means[uid].append(max_item_mean)
        self.As[uid] += max_item_mean[:,None].dot(max_item_mean[None,:])


class GLM_UCBInit(GLM_UCB):
    def __init__(self, init,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init = init
        self.parameters.extend(['init'])

    def train(self,train_dataset):
        super().train(train_dataset)

        if self.init == 'entropy':
            items_entropy = interactors.Entropy.get_items_entropy(self.train_consumption_matrix)
            self.items_bias = items_entropy
        elif self.init == 'popularity':
            items_popularity = interactors.MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)
            self.items_bias = items_popularity
        elif self.init == 'logpopent':
            items_entropy = interactors.Entropy.get_items_entropy(self.train_consumption_matrix)
            items_popularity = interactors.MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)
            self.items_bias = interactors.LogPopEnt.get_items_logpopent(items_popularity,items_entropy)
        elif self.init == 'rand_popularity':
            items_popularity = interactors.MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)
            items_popularity[np.argsort(items_popularity)[::-1][100:]] = 0
            self.items_bias = items_popularity
        elif self.init == 'random':
            self.items_bias = np.random.rand(self.train_dataset.num_total_items)

        self.items_bias = self.items_bias - np.min(self.items_bias)
        self.items_bias = self.items_bias/np.max(self.items_bias)

        assert(self.items_bias.min() >= 0 and np.isclose(self.items_bias.max(), 1))

        res=scipy.optimize.minimize(lambda x,items_means,items_bias: np.sum((items_bias - x @ items_means.T)**2),
                                    np.ones(self.num_latent_factors),
                                    args=(self.items_means,self.items_bias),
                                    method='BFGS',
                                    )
        self.initial_b = res.x 

        print(np.corrcoef(self.items_bias,self.initial_b @ self.items_means.T)[0,1])

        self.bs = defaultdict(lambda: self.initial_b.copy())

class GLM_UCBEntropy(GLM_UCBInit):
    def __init__(self,*args, **kwargs):
        super().__init__(init='entropy',*args, **kwargs)

class GLM_UCBPopularity(GLM_UCBInit):
    def __init__(self,*args, **kwargs):
        super().__init__(init='popularity',*args, **kwargs)

class GLM_UCBRandPopularity(GLM_UCBInit):
    def __init__(self,*args, **kwargs):
        super().__init__(init='rand_popularity',*args, **kwargs)

class GLM_UCBRandom(GLM_UCBInit):
    def __init__(self,*args, **kwargs):
        super().__init__(init='random',*args, **kwargs)

class GLM_UCBLogPopEnt(GLM_UCBInit):
    def __init__(self,*args, **kwargs):
        super().__init__(init='logpopent',*args, **kwargs)
