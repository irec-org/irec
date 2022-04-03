import numpy as np
from tqdm import tqdm, trange
from .ExperimentalValueFunction import ExperimentalValueFunction
import os
import random
import scipy.stats
from collections import defaultdict


class ThompsonSampling(ExperimentalValueFunction):
    """Thompson Sampling.
    
    A basic item-oriented bandit algorithm that follows a Gaussian distribution
    of items and users to perform the prediction rule based on their samples [1]_.


    References
    ----------
    .. [1] Chapelle, Olivier, and Lihong Li. "An empirical evaluation of thompson sampling."
        Advances in neural information processing systems 24 (2011): 2249-2257.   
    """
    def __init__(self, alpha_0, beta_0, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
            alpha_0:
            beta_0:
        """
        super().__init__(*args, **kwargs)
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0


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
        self.num_total_items = self.train_dataset.num_total_items

        self.alphas = np.ones(self.num_total_items) * self.alpha_0
        self.betas = np.ones(self.num_total_items) * self.beta_0

        for i in range(self.train_dataset.data.shape[0]):
            uid = int(self.train_dataset.data[i, 0])
            item = int(self.train_dataset.data[i, 1])
            reward = self.train_dataset.data[i, 2]
            # self.update(uid, item, reward, None)
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
        items_score = np.random.beta(self.alphas[candidate_items],
                                     self.betas[candidate_items])
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
        reward = 1 if (reward >= self.train_dataset.mean_rating) else 0
        self.alphas[item] += reward
        self.betas[item] += 1 - reward
