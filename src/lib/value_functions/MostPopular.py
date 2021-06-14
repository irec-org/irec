import numpy as np
from tqdm import tqdm
from .ExperimentalValueFunction import ExperimentalValueFunction
from .MostPopular import *
import matplotlib.pyplot as plt
import os
import scipy
from collections import defaultdict
import random


class MostPopular(ExperimentalValueFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_popularity(consumption_matrix, normalize=True):
        lowest_value = np.min(consumption_matrix)
        if not isinstance(consumption_matrix, scipy.sparse.spmatrix):
            items_popularity = np.count_nonzero(
                consumption_matrix > lowest_value, axis=0)
        else:
            items_popularity = np.array(
                np.sum(consumption_matrix > lowest_value, axis=0)).flatten()

        if normalize:
            items_popularity = items_popularity / consumption_matrix.shape[0]

        return items_popularity

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.num_total_items = self.train_dataset.num_total_items
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (self.train_dataset.data[:, 2],
             (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1])),
            (self.train_dataset.num_total_users,
             self.train_dataset.num_total_items))
        self.null_value = np.min(self.train_consumption_matrix)
        self.items_popularity = self.get_items_popularity(
            self.train_consumption_matrix, normalize=False)

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = self.items_popularity[candidate_items]
        return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        if reward != self.null_value:
            self.items_popularity[item] += 1
