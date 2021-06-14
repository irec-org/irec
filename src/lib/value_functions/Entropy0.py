import numpy as np
from tqdm import tqdm
from .ExperimentalValueFunction import ExperimentalValueFunction
import matplotlib.pyplot as plt
import os
import scipy.sparse

from .MostPopular import *


class Entropy0(ExperimentalValueFunction):
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
        # mask = np.ones(consumption_matrix.shape[0], dtype=bool)
        # mask[test_uids] = 0
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

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = self.items_entropy[candidate_items]
        return items_score, None
        # fig, ax = plt.subplots()
        # ax.hist(items_entropy,color='k')
        # ax.set_xlabel("Entropy0")
        # ax.set_ylabel("#Items")
        # fig.savefig(os.path.join(self.DIRS['img'],"entropy0_"+self.get_id()+".png"))

        # top_iids = list(reversed(np.argsort(items_entropy)))
