import numpy as np
from tqdm import tqdm
from .ExperimentalValueFunction import ExperimentalValueFunction
import os
import random
import scipy.stats
from collections import defaultdict


class EGreedy(ExperimentalValueFunction):
    """EGreedy.
    
    In general, ε-Greedy models the problem based on an ε diversification parameter to perform random actions [1]_.
    
    References
    ----------
    .. [1] Auer, P., Cesa-Bianchi, N. & Fischer, P. Finite-time Analysis of the 
       Multiarmed Bandit Problem. Machine Learning 47, 235–256 (2002).  
    """
    def __init__(self, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
        """
        super().__init__(*args, **kwargs)

    def reset(self, observation):
        """reset.

        Args:
            observation: 
        """
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (self.train_dataset.data[:, 2],
             (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1])),
            (self.train_dataset.num_total_users,
             self.train_dataset.num_total_items))
        self.num_total_items = self.train_consumption_matrix.shape[1]

        self.items_mean_values = np.zeros(self.num_total_items)
        self.items_count = np.zeros(self.num_total_items, dtype=int)

        for i in range(self.train_dataset.data.shape[0]):
            uid = int(self.train_dataset.data[i, 0])
            item = int(self.train_dataset.data[i, 1])
            reward = self.train_dataset.data[i, 2]
            self.update(None, (uid, item), reward, None)

    def action_estimates(self, candidate_actions):
        """action_estimates.

        Args:
            candidate_actions: (user id, candidate_items)

        Returns:
            numpy.ndarray:
        """
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        # if self.epsilon < np.random.rand():
        items_score = self.items_mean_values[candidate_items]
        # else:
        # items_score = np.random.rand(len(candidate_items))
        return items_score, None

    def update(self, observation, action, reward, info):
        """update.

        Args:
            observation:
            action: (user id, item)
            reward (float): reward
            info: 
        """
        uid = action[0]
        item = action[1]
        additional_data = info
        self.items_mean_values[item] = (
            self.items_mean_values[item] * self.items_count[item] +
            reward) / (self.items_count[item] + 1)
        self.items_count[item] += 1
