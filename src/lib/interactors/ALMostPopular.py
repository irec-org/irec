import numpy as np
from tqdm import tqdm
from .ExperimentalInteractor import ExperimentalInteractor
from .MostPopular import *
import matplotlib.pyplot as plt
import os
import scipy.sparse
from collections import defaultdict
import random
class ALMostPopular(ExperimentalInteractor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.num_items = self.train_dataset.num_items
        self.items_popularity = MostPopular.get_items_popularity(self.train_consumption_matrix, normalize=False)

    def predict(self,uid,candidate_items,num_req_items):
        items_score = self.items_popularity[candidate_items]
        return items_score, None
        # top_items = list(reversed(np.argsort(items_score)))[:self.interaction_size]

    def update(self,uid,item,reward,additional_data):
        self.items_popularity[item] += 1
