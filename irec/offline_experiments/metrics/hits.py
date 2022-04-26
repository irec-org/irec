import numpy as np
from collections import defaultdict
from .base import Metric
from typing import Any

np.seterr(all="raise")

class Hits(Metric):
    """Hits.

    Number of recommendations made successfully.
    (right predictions)
    """

    def __init__(self, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
        """
        super().__init__(*args, **kwargs)
        self.users_true_positive = defaultdict(int)

    def compute(self, uid: int):
        """compute.

        Args:
            uid (int): user id
        """
        return self.users_true_positive[uid]

    def update_recommendation(self, uid: int, item: int, reward: float):
        """update_recommendation.

        Args:
            uid (int): user id
            item (int): item id
            reward (float): reward
        """
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid] += 1
