import numpy as np
from collections import defaultdict
from .base import Metric
from typing import Any

np.seterr(all="raise")

class UsersCoverage(Metric):
    """Users Coverage.

    It represents the percentage of distinctusers that are interested
    in at least k items recommended (k ≥ 1).
    """

    def __init__(self, users_covered=defaultdict(bool), *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
            users_covered:
        """
        super().__init__(*args, **kwargs)
        self.users_covered = users_covered

    def compute(self, uid: int):
        """compute.

        Args:
            uid (int): user id
        """
        vals = np.array(list(self.users_covered.values()))
        return np.sum(vals) / len(vals)

    def update_recommendation(self, uid: int, item: int, reward: float):
        """update_consumption_history.

        Args:
            uid (int): user id
            item (int): item id
            reward (float): reward
        """
        if self.users_covered[uid] is False and self.relevance_evaluator.is_relevant(
            reward
        ):
            self.users_covered[uid] = True
            # else:
            # self.users_covered[uid] = False
