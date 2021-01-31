from .ExperimentalInteractor import *
import numpy as np
import random
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import mf
import ctypes
from .MFInteractor import MFInteractor

class LinEGreedy(MFInteractor):

    def __init__(self, epsilon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.parameters.extend(['epsilon'])
    
    def _init_items_weights(self):
        mf_model = mf.SVD()
        mf_model.fit(self.train_consumption_matrix)
        self.items_weights = mf_model.items_weights

    def train(self, train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (self.train_dataset.data[:, 2],
             (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1])),
            (self.train_dataset.num_total_users,
             self.train_dataset.num_total_items))
        self.num_total_items = self.train_dataset.num_total_items
        self._init_items_weights()

        self.bs = defaultdict(lambda: np.zeros(self.num_lat))
        self.As = defaultdict(lambda: self.init_A(self.num_lat))

    def init_A(self, num_lat):
        return np.eye(num_lat)

    def predict(self, uid, candidate_items, num_req_items):
        b = self.bs[uid]
        A = self.As[uid]

        mean = np.dot(np.linalg.inv(A), b)

        rand = np.random.rand(min(num_req_items, len(candidate_items)))
        rand = self.epsilon > rand

        cnz = np.count_nonzero(rand)
        if cnz == min(num_req_items, len(candidate_items)):
            items_score = np.random.rand(len(candidate_items))
        else:
            items_score = mean @ self.items_weights[candidate_items].T
            randind = random.sample(list(range(len(candidate_items))),
                                    k=np.count_nonzero(rand))
            items_score[randind] = np.inf
        return items_score, None

    def update(self, uid, item, reward, additional_data):
        b = self.bs[uid]
        A = self.As[uid]
        max_item_mean = self.items_weights[item]
        A += max_item_mean[:, None].dot(max_item_mean[None, :])
        b += reward * max_item_mean
