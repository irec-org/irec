import numpy as np
from collections import defaultdict
from typing import Any
from .base import Metric

np.seterr(all="raise")

class Recall(Metric):
    """Recall.

    Recall represents the probability that a relevant item will be selected.
    (true positive/false negative)
    """

    def __init__(self, users_false_negative, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
            users_false_negative:
        """

        super().__init__(*args, **kwargs)
        self.users_true_positive = defaultdict(int)
        self.users_false_negative = users_false_negative

    def compute(self, uid: int):
        """compute.

        Args:
            uid (int): user id
        """
        if self.users_true_positive[uid] == 0 and self.users_false_negative[uid] == 0:
            return 0
        return self.users_true_positive[uid] / self.users_false_negative[uid]

    def update_recommendation(self, uid: int, item: int, reward: float):
        """update_recommendation.

        Args:
            uid (int): user id
            item (int): item id
            reward (float): reward
        """
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid] += 1

