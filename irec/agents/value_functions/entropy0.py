import numpy as np
from .base import ValueFunction
import scipy.sparse

from .most_popular import *


class Entropy0(ValueFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def probabilities_entropy(probabilities):
        return -1 * np.sum(probabilities * np.log(probabilities))

    @staticmethod
    def values_entropy(values):
        unique, counts = np.unique(values, return_counts=True)
        values_probability = counts / np.sum(counts)
        return Entropy0.probabilities_entropy(values_probability)

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
            else:
                raise RuntimeError
            items_entropy[iid] = Entropy0.values_entropy(iid_ratings)
        return items_entropy

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

        self.items_entropy = self.get_items_entropy(
            self.train_consumption_matrix)

    def actions_estimate(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = self.items_entropy[candidate_items]
        return items_score, None