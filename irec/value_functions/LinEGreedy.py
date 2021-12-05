from typing import Any, DefaultDict
from .ExperimentalValueFunction import *
import numpy as np
import random
from tqdm import tqdm
from threadpoolctl import threadpool_limits
from collections import defaultdict
import mf
import ctypes
from .MFValueFunction import MFValueFunction
import scipy.sparse


class LinEGreedy(MFValueFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.epsilon = epsilon
        #

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (
                self.train_dataset.data[:, 2],
                (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1]),
            ),
            (self.train_dataset.num_total_users, self.train_dataset.num_total_items),
        )
        self.num_total_items = self.train_dataset.num_total_items
        mf_model = mf.SVD()
        mf_model.fit(self.train_consumption_matrix)
        self.items_weights = mf_model.items_weights

        self.bs: DefaultDict[Any, Any] = defaultdict(lambda: np.ones(self.num_lat))
        self.As: DefaultDict[Any, Any] = defaultdict(lambda: np.eye(self.num_lat))

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        b = self.bs[uid]
        A = self.As[uid]

        mean = np.dot(np.linalg.inv(A), b)
        items_score = mean @ self.items_weights[candidate_items].T

        # rand = np.random.rand(min(num_req_items, len(candidate_items)))
        # rand = self.epsilon > rand

        # cnz = np.count_nonzero(rand)
        # if cnz == min(num_req_items, len(candidate_items)):
        # items_score = np.random.rand(len(candidate_items))
        # else:
        # items_score = mean @ self.items_weights[candidate_items].T
        # randind = random.sample(list(range(len(candidate_items))),
        # k=np.count_nonzero(rand))
        # items_score[randind] = np.inf
        return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        max_item_mean = self.items_weights[item]
        self.As[uid] += max_item_mean[:, None].dot(max_item_mean[None, :])
        self.bs[uid] += reward * max_item_mean
