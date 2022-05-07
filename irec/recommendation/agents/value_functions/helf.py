import numpy as np

from entropy import Entropy
from . import most_popular
import scipy.stats
from .base import ValueFunction


class HELF(ValueFunction):

    """HELF

    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_helf(items_popularity, items_entropy, num_total_users):

        a = np.ma.log(items_popularity).filled(0) / np.log(num_total_users)
        b = items_entropy / np.max(items_entropy)
        print(np.sort(b))
        print(b.min())
        print(a.min())
        print(np.sort(a))
        np.seterr('warn')
        items_helf = 2 * a * b / (a + b)
        items_helf[np.isnan(items_helf)] = 0
        return items_helf

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
        num_train_users = len(self.train_dataset.uids)
        items_entropy = Entropy.get_items_entropy(
            self.train_consumption_matrix)
        items_popularity = most_popular.get_items_popularity(
            self.train_consumption_matrix, normalize=False)
        self.items_logpopent = HELF.get_items_helf(items_popularity,
                                                   items_entropy,
                                                   num_train_users)

    def actions_estimate(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = self.items_logpopent[candidate_items]
        return items_score, None
