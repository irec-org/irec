import numpy as np
from tqdm import tqdm
from .ExperimentalValueFunction import ExperimentalValueFunction
import matplotlib.pyplot as plt
import os
import scipy.sparse


class Entropy(ExperimentalValueFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def probabilities_entropy(probabilities):
        try:
            return -1 * np.sum(probabilities * np.log(probabilities))
        except:
            return 0

    @staticmethod
    def values_entropy(values):
        unique, counts = np.unique(values, return_counts=True)
        values_probability = counts / np.sum(counts)
        return Entropy.probabilities_entropy(values_probability)

    @staticmethod
    def get_items_entropy(consumption_matrix):
        lowest_value = np.min(consumption_matrix)
        items_entropy = np.zeros(consumption_matrix.shape[1])
        is_spmatrix = isinstance(consumption_matrix, scipy.sparse.spmatrix)
        if is_spmatrix:
            consumption_matrix = scipy.sparse.csc_matrix(consumption_matrix)
        for iid in range(consumption_matrix.shape[1]):
            if is_spmatrix:
                iid_ratings = consumption_matrix[:, iid].A.flatten()
                iid_ratings = iid_ratings[iid_ratings > lowest_value]
            else:
                iid_ratings = consumption_matrix[:, iid]
                iid_ratings = iid_ratings[iid_ratings > lowest_value]
            items_entropy[iid] = Entropy.values_entropy(iid_ratings)
        return items_entropy

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.num_total_items = self.train_dataset.num_total_items
        self.unique_values = self.train_dataset.rate_domain
        self.num_unique_values = len(self.unique_values)
        self.items_ratings = np.zeros(
            (self.num_total_items, self.num_unique_values))
        self.unique_values_ids = dict(
            zip(self.unique_values, list(range(self.num_unique_values))))
        self.items_num_total_ratings = np.zeros(self.num_total_items)
        for uid, iid, reward, *rest in self.train_dataset.data:
            self.items_ratings[int(iid), self.unique_values_ids[reward]] += 1
            self.items_num_total_ratings[int(iid)] += 1

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = [
            self.probabilities_entropy(self.items_ratings[iid] /
                                       np.sum(self.items_ratings[iid]))
            if self.items_num_total_ratings[iid] > 0 else 0
            for iid in candidate_items
        ]
        return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        if reward in self.unique_values_ids:
            self.items_ratings[item, self.unique_values_ids[reward]] += 1
            self.items_num_total_ratings[item] += 1
