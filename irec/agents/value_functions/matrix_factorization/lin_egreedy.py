from typing import Any, DefaultDict
from ..experimental.experimental_valueFunction import *
import numpy as np
from collections import defaultdict
from . import mf
from .mf_value_function import MFValueFunction
import scipy.sparse


class LinEGreedy(MFValueFunction):

    """LinEGreedy

    A linear exploitation of the items latent factors defined by a SVD
    formulation that also explore random items with probability ε
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        max_item_mean = self.items_weights[item]
        self.As[uid] += max_item_mean[:, None].dot(max_item_mean[None, :])
        self.bs[uid] += reward * max_item_mean
