import numpy as np
from tqdm import tqdm
from .Interactor import *
from .Entropy import *
import matplotlib.pyplot as plt
import os
import scipy.sparse
from collections import defaultdict
import random
class ALEntropy(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self,train_data):
        super().train(train_data)
        self.train_consumption_matrix = scipy.sparse.csr_matrix((train_data[2],(train_data[0],train_data[1])))
        # uids = self.test_users
        self.num_items = self.train_consumption_matrix.shape[1]
        self.unique_values = np.unique(np.append(self.train_consumption_matrix.data,self.test_consumption_matrix.data))
        self.num_unique_values = len(unique_values)
        self.items_ratings = np.zeros((self.num_items,self.num_unique_values))
        self.unique_values_ids = dict(zip(unique_values,list(range(num_unique_values))))


    def predict(self,uid,candidate_items):
        items_score =  [Entropy.values_entropy(self.items_ratings[iid])
                        for iid
                        in candidate_items]
        return items_score
        # top_item = list(reversed(np.argsort(items_score)))[0]
        # best_item = candidate_items[top_item]

    def update(self,uid,item,reward):
        items_ratings[item,self.unique_values_ids[reward]] = reward
