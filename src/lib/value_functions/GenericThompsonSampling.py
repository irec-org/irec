import numpy as np
from numpy.lib.npyio import save
from tqdm import tqdm, trange
from . import ValueFunction, ExperimentalValueFunction
import os
import random
import scipy.stats
from collections import defaultdict


class GenericThompsonSampling(ExperimentalValueFunction):

    def __init__(self, alpha_0, beta_0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        # self.parameters.extend(['alpha_0', 'beta_0'])

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        # self.train_dataset = train_dataset
        # self.train_consumption_matrix = scipy.sparse.csr_matrix(
        # (self.train_dataset.data[:, 2],
        # (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1])),
        # (self.train_dataset.num_total_users,
        # self.train_dataset.num_total_items))
        # self.num_total_items = self.train_dataset.num_total_items

        self.alphas = defaultdict(lambda : self.alpha_0)
        self.betas = defaultdict(lambda:self.beta_0)

        # for i in range(self.train_dataset.data.shape[0]):
            # uid = int(self.train_dataset.data[i, 0])
            # item = int(self.train_dataset.data[i, 1])
            # reward = self.train_dataset.data[i, 2]
            # # self.action_estimates()
            # # self.update(uid, item, reward, None)
            # self.update(None, (uid, item), reward, None)

    def action_estimates(self, candidate_actions):
        items_score = np.random.beta([self.alphas[ca] for ca in candidate_actions],
                                     [self.betas[ca] for ca in candidate_actions])
        return items_score, None

    def update(self, observation, action, reward, info):
        # additional_data = info
        reward = 1 if (reward >= 4) else 0
        self.alphas[action] += reward
        self.betas[action] += 1 - reward
