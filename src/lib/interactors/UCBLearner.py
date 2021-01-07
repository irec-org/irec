from .ExperimentalInteractor import ExperimentalInteractor
import numpy as np
import random
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
import ctypes
from .Entropy import Entropy
from .MostPopular import MostPopular
from .LogPopEnt import LogPopEnt
from .PopPlusEnt import *

class UCBLearner(ExperimentalInteractor):
    def __init__(self, stop=14, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = stop

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[2],(self.train_dataset.data[0],self.train_dataset.data[1])),(self.train_dataset.users_num,self.train_dataset.items_num))
        self.num_items = self.train_dataset.num_items

        items_entropy = Entropy.get_items_entropy(self.train_consumption_matrix)
        items_popularity = MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)
        self.items_bias= LogPopEnt.get_items_logpopent(items_popularity,items_entropy)

        mf_model = mf.SVD()
        mf_model.fit(self.train_consumption_matrix)
        self.items_weights = mf_model.items_weights
        self.num_latent_factors = len(self.items_weights[0])

        self.I = np.eye(len(self.items_weights[0]))
        bs = defaultdict(lambda: np.zeros(self.num_latent_factors))
        As = defaultdict(lambda: self.I.copy())
        self.users_nb_items = defaultdict(lambda: 0)

    @staticmethod
    def discount_bias(num_items,stop):
        limit = pow(2,stop)/100
        return pow(2,min(stop,num_items))/limit

    def predict(self,uid,candidate_items,num_req_items):
        b = bs[uid]
        A = As[uid]
        items_bias = self.items_bias
        mean = np.dot(np.linalg.inv(A),b)
        pred_rule = mean @ self.items_weights[user_candidate_items].T
        current_bias = items_bias[user_candidate_items] * max(1, np.max(pred_rule))
        bias = current_bias - (current_bias * self.discount_bias(self.users_nb_items[uid],self.stop)/100)
        bias[bias<0] = 0
        items_score = pred_rule + bias)[::-1]
        return items_score

    def update(self,uid,item,reward,additional_data):
        max_item_weight = self.items_weights[item]
        A += max_item_weight[:,None].dot(max_item_weight[None,:])
        b += reward*max_item_weight
        self.users_nb_items[uid] += 1
