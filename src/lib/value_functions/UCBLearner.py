from .ExperimentalValueFunction import ExperimentalValueFunction
import numpy as np
import random
from tqdm import tqdm
#import util
from threadpoolctl import threadpool_limits
import ctypes
from .Entropy import Entropy
from .MostPopular import MostPopular
from .LogPopEnt import LogPopEnt
from .PopPlusEnt import *
from .MFValueFunction import MFValueFunction


class UCBLearner(MFValueFunction):
    def __init__(self, stop=14, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = stop


    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (self.train_dataset.data[:, 2],
             (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1])),
            (self.train_dataset.num_total_users,
             self.train_dataset.num_total_items))
        self.num_total_items = self.train_dataset.num_total_items

        items_entropy = Entropy.get_items_entropy(
            self.train_consumption_matrix)
        items_popularity = MostPopular.get_items_popularity(
            self.train_consumption_matrix, normalize=False)
        self.items_bias = LogPopEnt.get_items_logpopent(
            items_popularity, items_entropy)

        mf_model = mf.SVD(num_lat=self.num_lat)
        mf_model.fit(self.train_consumption_matrix)
        self.items_weights = mf_model.items_weights
        self.num_latent_factors = len(self.items_weights[0])

        self.I = np.eye(len(self.items_weights[0]))
        self.bs = defaultdict(lambda: np.zeros(self.num_latent_factors))
        self.As = defaultdict(lambda: self.I.copy())
        self.users_nb_items = defaultdict(lambda: 0)

    @staticmethod
    def discount_bias(num_total_items, stop):
        limit = pow(2, stop) / 100
        return pow(2, min(stop, num_total_items)) / limit

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        b = self.bs[uid]
        A = self.As[uid]
        items_bias = self.items_bias
        mean = np.dot(np.linalg.inv(A), b)
        pred_rule = mean @ self.items_weights[user_candidate_items].T
        current_bias = items_bias[user_candidate_items] * max(
            1, np.max(pred_rule))
        bias = current_bias - (current_bias * self.discount_bias(
            self.users_nb_items[uid], self.stop) / 100)
        bias[bias < 0] = 0
        items_score = (pred_rule + bias)[::-1]
        return items_score

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        b = self.bs[uid]
        A = self.As[uid]
        max_item_weight = self.items_weights[item]
        A += max_item_weight[:, None].dot(max_item_weight[None, :])
        b += reward * max_item_weight
        self.users_nb_items[uid] += 1
