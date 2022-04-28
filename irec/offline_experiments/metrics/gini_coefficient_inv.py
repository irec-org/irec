import numpy as np
from collections import defaultdict
from .base import Metric
from typing import Any

np.seterr(all="raise")


class GiniCoefficientInv(Metric):
    """GiniCoefficientInv.

    desc
    """

    def __init__(self, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
            users_covered:
        """
        super().__init__(*args, **kwargs)
        self.items_frequency = defaultdict(int)
        for item in np.unique(self.ground_truth_dataset.data[:, 1]):
            self.items_frequency[item]
        self.is_computation_updated = False
        self.computation_cache = None

    def compute(self, uid: int):
        """compute.

        Args:
            uid (int): user id
        """
        if self.is_computation_updated is False:
            self.is_computation_updated = True
            x = np.array(list(self.items_frequency.values()))
            diff_sum = 0
            for i, xi in enumerate(x[:-1], 1):
                diff_sum += np.sum(np.abs(xi - x[i:]))
            self.computation_cache = diff_sum / (len(x) ** 2 * np.mean(x))
        return 1 - self.computation_cache

    def update_recommendation(self, uid: int, item: int, reward: float):
        """update_recommendation.

        Args:
            uid (int): user id
            item (int): item id
            reward (float): reward
        """
        self.items_frequency[item] += 1
        if self.is_computation_updated:
            self.is_computation_updated = False
