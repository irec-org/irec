import numpy as np
from tqdm import tqdm
from . import Interactor, ExperimentalInteractor
import os
import random
import scipy.stats
from collections import defaultdict

class EGreedy(ExperimentalInteractor):
    def __init__(self, epsilon=0.1,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.parameters.extend(['epsilon'])

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[:,2],(self.train_dataset.data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.num_total_users,self.train_dataset.num_total_items))
        self.num_total_items = self.train_consumption_matrix.shape[1]
        
        self.items_mean_values = np.zeros(self.num_total_items)
        self.items_count = np.zeros(self.num_total_items,dtype=int)

        for i in range(self.train_dataset.data.shape[0]):
            uid = int(self.train_dataset.data[i,0])
            item = int(self.train_dataset.data[i,1])
            reward = self.train_dataset.data[i,2]
            self.update(uid,item,reward,None)
            self.increment_time()

    def predict(self,uid,candidate_items,num_req_items):
        if self.epsilon < np.random.rand():
            items_score = self.items_mean_values[candidate_items]
        else:
            items_score = np.random.rand(len(candidate_items))
        return items_score, None
                
    def update(self,uid,item,reward,additional_data):
        self.items_mean_values[item] = (self.items_mean_values[item]*self.items_count[item]+reward)/(self.items_count[item] + 1)
        self.items_count[item] += 1
