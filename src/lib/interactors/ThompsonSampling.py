import numpy as np
from tqdm import tqdm, trange
from . import Interactor, ExperimentalInteractor
import os
import random
import scipy.stats
from collections import defaultdict
class ThompsonSampling(ExperimentalInteractor):
    def __init__(self,alpha_0, beta_0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.parameters.extend(['alpha_0','beta_0'])

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[:,2],(self.train_dataset.data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.num_total_users,self.train_dataset.num_total_items))
        self.num_total_items = self.train_dataset.num_total_items

        self.alphas = np.ones(self.num_total_items)*self.alpha_0
        self.betas = np.ones(self.num_total_items)*self.beta_0

        a = np.sum(self.train_consumption_matrix>=self.train_dataset.mean_rating,
                                        axis=0).A.flatten()
        self.alphas += a
        self.betas += self.train_consumption_matrix.shape[0] - a

    def predict(self,uid,candidate_items,num_req_items):
        items_score = np.random.beta(self.alphas[candidate_items],
                                     self.betas[candidate_items])
        return items_score, None

    def update(self,uid,item,reward,additional_data):
        reward = reward
        reward = 1 if reward >= self.train_dataset.mean_rating else 0
        self.alphas[item] += reward
        self.betas[item] += 1-reward
