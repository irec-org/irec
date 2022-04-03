import numpy as np
from tqdm import tqdm
from . import Entropy, MostPopular, LogPopEnt, ExperimentalValueFunction
import matplotlib.pyplot as plt
import scipy.stats
import os


class LogPopEnt(ExperimentalValueFunction.ExperimentalValueFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_logpopent(items_popularity, items_entropy, k=None):
        if k is not None:
            items_logpopent = (items_entropy ** k) * np.ma.log(items_popularity).filled(
                0
            ) ** (1 - k)
        else:
            items_logpopent = items_entropy * np.ma.log(items_popularity).filled(0)
        return np.dot(items_logpopent, 1 / np.max(items_logpopent))

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

        items_entropy = Entropy.Entropy.get_items_entropy(self.train_consumption_matrix)
        items_popularity = MostPopular.MostPopular.get_items_popularity(
            self.train_consumption_matrix, normalize=False
        )
        self.items_logpopent = LogPopEnt.LogPopEnt.get_items_logpopent(
            items_popularity, items_entropy
        )

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = self.items_logpopent[candidate_items]
        return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        pass
