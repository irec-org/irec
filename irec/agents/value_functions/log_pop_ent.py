import numpy as np

from . import log_pop_ent

from . import entropy
from . import most_popular
import scipy.stats
from .base import ValueFunction


class LogPopEnt(ValueFunction):


    """LogPopEnt

    It combines popularity and entropy to identify potentially relevant items
    that also have the ability to add more knowledge to the system. As these 
    concepts are not strongly correlated, it is possible to achieve this 
    combination through a linear combination of the popularity 'p' of an item 
    i by its entropy ε: score(i) = log('p'i) · εi.
    
    References
    ----------
    .. Mehdi Elahi, Francesco Ricci, and Neil Rubens. 2016. A survey of active learning
       in collaborative filtering recommender systems. Computer Science Review 20 (2016), 29–50. 
    """


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_logpopent(items_popularity, items_entropy, k=None):
        if k is not None:
            items_logpopent = (items_entropy ** k) * np.ma.log(items_popularity).filled(
                0
            ) ** (1 - k)
        else:
            items_logpopent = items_entropy * np.ma.log(items_popularity).filled(0)
        return np.dot(items_logpopent, 1 / np.max(items_logpopent))

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (
                self.train_dataset.data[:, 2],
                (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1]),
            ),
            (self.train_dataset.num_total_users, self.train_dataset.num_total_items),
        )

        items_entropy = entropy.Entropy.get_items_entropy(self.train_consumption_matrix)
        items_popularity = most_popular.MostPopular.get_items_popularity(
            self.train_consumption_matrix, normalize=False
        )
        self.items_logpopent = log_pop_ent.LogPopEnt.get_items_logpopent(
            items_popularity, items_entropy
        )

    def actions_estimate(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = self.items_logpopent[candidate_items]
        return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        pass
