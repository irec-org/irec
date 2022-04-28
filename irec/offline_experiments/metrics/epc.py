import numpy as np
from collections import defaultdict
from .base import Metric

np.seterr(all="raise")

class EPC(Metric):
    """Expected Popularity Complement.

    EPC is a metric that measures the ability of a system to recommend
    relevant items that reside in the long-tail.
    """

    def __init__(self, items_normalized_popularity, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
            items_normalized_popularity
        """
        super().__init__(*args, **kwargs)
        self.users_num_items_recommended = defaultdict(int)
        self.users_prob_not_seen_cumulated = defaultdict(float)
        self.items_normalized_popularity = items_normalized_popularity

    def compute(self, uid: int):
        """compute.

        Args:
            uid (int): user id
        """
        C_2 = 1.0 / self.users_num_items_recommended[uid]
        sum_2 = self.users_prob_not_seen_cumulated[uid]
        EPC = C_2 * sum_2
        return EPC

    def update_recommendation(self, uid: int, item: int, reward: float):
        """update_recommendation.

        Args:
            uid (int): user id
            item (int): item id
            reward (float): reward
        """
        self.users_num_items_recommended[uid] += 1
        probability_seen = self.items_normalized_popularity[item]
        self.users_prob_not_seen_cumulated[uid] += 1 - probability_seen
