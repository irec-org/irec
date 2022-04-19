import numpy as np
from . import entropy, most_popular
import scipy.stats
from .experimental_valueFunction import ExperimentalValueFunction


class PopPlusEnt(ExperimentalValueFunction):

    """PopPlusEnt
    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_popplusent(items_popularity, items_entropy, log=False):
        if not log:
            items_popplusent = items_entropy / np.max(
                items_entropy) + items_popularity / np.max(items_popularity)
        else:
            items_popplusent = items_entropy / np.max(
                items_entropy) + np.ma.log(items_popularity).filled(
                    0) / np.max(np.ma.log(items_popularity).filled(0))
        return items_popplusent / np.max(items_popplusent)

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (self.train_dataset.data[:, 2],
             (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1])),
            (self.train_dataset.num_total_users,
             self.train_dataset.num_total_items))

        items_entropy = entropy.get_items_entropy(
            self.train_consumption_matrix)
        items_popularity = most_popular.get_items_popularity(
            self.train_consumption_matrix, normalize=False)
        self.items_popplusent = PopPlusEnt.get_items_popplusent(
            items_popularity, items_entropy)

    def actions_estimate(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = self.items_popplusent[candidate_items]
        return items_score, None