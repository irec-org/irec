import numpy as np
from collections import defaultdict
from .base import Metric
from typing import Any

np.seterr(all="raise")

class Precision(Metric):
    """Precision.

    Precision is defined as the percentage of predictions we get right.
    (true positive)/(total predictions).
    """

    def __init__(self, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
        """
        super().__init__(*args, **kwargs)
        self.users_true_positive = defaultdict(int)
        self.users_false_positive = defaultdict(int)

    def compute(self, uid: int):
        """compute.

        Args:
            uid (int): user id
        """
        if self.users_true_positive[uid] == 0 and self.users_false_positive[uid] == 0:
            return 0
        return self.users_true_positive[uid] / (
            self.users_true_positive[uid] + self.users_false_positive[uid]
        )

    def update_recommendation(self, uid: int, item: int, reward: float):
        """update_recommendation.

        Args:
            uid (int): user id
            item (int): item id
            reward (float): reward
        """
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid] += 1
        else:
            self.users_false_positive[uid] += 1
