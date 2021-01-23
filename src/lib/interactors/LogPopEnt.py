import numpy as np
from tqdm import tqdm
from . import Interactor, Entropy, MostPopular, LogPopEnt, ExperimentalInteractor
import matplotlib.pyplot as plt
import scipy.stats
import os

class LogPopEnt(ExperimentalInteractor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_logpopent(items_popularity,items_entropy):
        items_logpopent = items_entropy * np.ma.log(items_popularity).filled(0)
        return np.dot(items_logpopent,1/np.max(items_logpopent))

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[:,2],(self.train_dataset.data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.num_total_users,self.train_dataset.num_total_items))

        items_entropy = Entropy.get_items_entropy(self.train_consumption_matrix)
        items_popularity = MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)
        self.items_logpopent = LogPopEnt.get_items_logpopent(items_popularity,items_entropy)

    def predict(self,uid,candidate_items,num_req_items):
        items_score = self.items_logpopent[candidate_items]
        return items_score, None

    def update(self,uid,item,reward,additional_data):
        pass
