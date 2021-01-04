import numpy as np
from tqdm import tqdm
from .Interactor import Interactor
from .MostPopular import *
import matplotlib.pyplot as plt
import os
import scipy.sparse
from collections import defaultdict
import random
class ALMostPopular(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[2],(self.train_dataset.data[0],self.train_dataset.data[1])),(self.train_dataset.users_num,self.train_dataset.items_num))
        self.num_items = self.train_consumption_matrix.shape[1]
        self.items_popularity = MostPopular.get_items_popularity(self.train_consumption_matrix, normalize=False)

    def predict(self,uid,candidate_items):
        items_score = self.items_popularity[candidate_items]
        return items_score, None
        # top_items = list(reversed(np.argsort(items_score)))[:self.interaction_size]

    def update(self,uid,item,reward,additional_data):
        self.items_popularity[item] += 1
