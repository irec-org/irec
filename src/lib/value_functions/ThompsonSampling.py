import numpy as np
from tqdm import tqdm, trange
from . import ValueFunction, ExperimentalValueFunction
import os
import random
import scipy.stats
from collections import defaultdict


class ThompsonSampling(ExperimentalValueFunction):
    def __init__(self, alpha_0, beta_0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0


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

        self.alphas = np.ones(self.num_total_items) * self.alpha_0
        self.betas = np.ones(self.num_total_items) * self.beta_0

        for i in range(self.train_dataset.data.shape[0]):
            uid = int(self.train_dataset.data[i, 0])
            item = int(self.train_dataset.data[i, 1])
            reward = self.train_dataset.data[i, 2]
            # self.update(uid, item, reward, None)
            self.update(None, (uid, item), reward, None)

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = np.random.beta(self.alphas[candidate_items],
                                     self.betas[candidate_items])
        return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        reward = 1 if (reward >= self.train_dataset.mean_rating) else 0
        self.alphas[item] += reward
        self.betas[item] += 1 - reward
