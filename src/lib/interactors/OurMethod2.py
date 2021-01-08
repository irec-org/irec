import numpy as np
from tqdm import tqdm
import util
import interactors
from threadpoolctl import threadpool_limits
import ctypes
import scipy.spatial
import matplotlib.pyplot as plt
import os
import pickle
import sklearn
import scipy.optimize
class OurMethod2(interactors.ExperimentalInteractor):
    def __init__(self, alpha=1.0,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters['alpha'] = alpha

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[2],(self.train_dataset.data[0],self.train_dataset.data[1])),(self.train_dataset.users_num,self.train_dataset.items_num))
        self.num_items = self.train_dataset.num_items

        mf_model = mf.SVD()
        mf_model.fit(self.train_consumption_matrix)
        self.items_weights = mf_model.items_weights
        self.num_latent_factors = len(self.items_weights[0])

        items_entropy = interactors.Entropy.get_items_entropy(self.train_consumption_matrix)
        items_popularity = interactors.MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)
        # self.items_bias = interactors.PPELPE.get_items_ppelpe(items_popularity,items_entropy)
        self.items_bias = interactors.LogPopEnt.get_items_logpopent(items_popularity,items_entropy)
        assert(self.items_bias.min() >= 0 and self.items_bias.max() == 1)

        # regression_model = sklearn.linear_model.LinearRegression()
        res=scipy.optimize.minimize(lambda x,items_weights,items_bias: np.linalg.norm(items_bias - x @ items_weights.T),
                                    np.ones(num_lat),
                                    args=(self.items_weights,self.items_bias))
        self.initial_b = res.x 

        print(np.corrcoef(self.items_bias,self.initial_b @ items_weights.T)[0,1])

        self.I = np.eye(len(self.items_weights[0]))
        bs = defaultdict(lambda: np.zeros(self.num_latent_factors))
        As = defaultdict(lambda: self.initial_b.copy())

    def predict(self,uid,candidate_items,num_req_items):
        b = bs[uid]
        A = As[uid]
        user_latent_factors = np.dot(np.linalg.inv(A),b)
        items_uncertainty = np.sqrt(np.sum(self.items_weights[user_candidate_items].dot(np.linalg.inv(A)) * self.items_weights[user_candidate_items],axis=1))
        items_user_similarity = user_latent_factors @ self.items_weights[user_candidate_items].T
        user_model_items_score = items_user_similarity + self.parameters['alpha']*items_uncertainty
        items_score = user_model_items_score
        return items_score, None

    def update(self,uid,item,reward,additional_data):
        max_item_latent_factors = self.items_weights[item]
        b = bs[uid]
        A = As[uid]
        A += max_item_latent_factors[:,None].dot(max_item_latent_factors[None,:])
        b += reward*max_item_latent_factors
