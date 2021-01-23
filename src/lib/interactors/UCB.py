import numpy as np
from tqdm import tqdm
from . import Interactor, ExperimentalInteractor
import os
import random
import scipy.stats
from collections import defaultdict


class UCB(ExperimentalInteractor):
    def __init__(self,c,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.parameters.extend(['c'])

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[:,2],(self.train_dataset.data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.num_total_users,self.train_dataset.num_total_items))
        self.num_total_items = self.train_dataset.num_total_items
        
        self.items_mean_values = np.zeros(self.num_total_items,dtype=np.float128)
        self.items_count = np.zeros(self.num_total_items,dtype=int)

        self.t = 1

        # users_num_interactions = defaultdict(int)
        # available_users = set(uids)

        for i in range(self.train_dataset.data.shape[0]):
            uid = self.train_dataset.data[i,0]
            item = self.train_dataset.data[i,1]
            reward = self.train_dataset.data[i,2]
            self.update(uid,item,reward,None)
            self.increment_time()
        # mask = np.ones(self.train_consumption_matrix.shape[0], dtype=bool)
        # mask[uids] = 0
        # self.items_mean_values = np.mean(self.train_consumption_matrix[mask],axis=0).A.flatten()
        # self.items_count += self.train_consumption_matrix[mask].shape[0]
        # self.t += np.prod(self.train_consumption_matrix[mask].shape)

    def predict(self,uid,candidate_items,num_req_items):
        with np.errstate(divide='ignore'):
            items_uncertainty = self.c*np.sqrt(2*np.log(self.t)/self.items_count[candidate_items])
            items_score = self.items_mean_values[candidate_items]+items_uncertainty
        return items_score, None

    def update(self,uid,item,reward,additional_data):
        item = int(item)
        self.items_mean_values[item] = (self.items_mean_values[item]*self.items_count[item]+reward)/(self.items_count[item] + 1)
        self.items_count[item] += 1
