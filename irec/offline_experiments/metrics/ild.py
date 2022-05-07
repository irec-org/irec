import numpy as np
from collections import defaultdict
from .base import Metric
from typing import Any

np.seterr(all="raise")


class ILD(Metric):
    """Intra-List Diversity.

    This is used to measure the diversity of an individual user’s recommendations and quantifies user-novelty.
    """

    def __init__(self, items_distance, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
            items_distance:
        """
        super().__init__(*args, **kwargs)
        self.items_distance = items_distance
        self.users_items_recommended = defaultdict(list)
        self.users_local_ild = defaultdict(float)

    def compute(self, uid: int):
        """compute.

        Args:
            uid (int): user id
        """
        user_num_items_recommended = len(self.users_items_recommended[uid])
        if user_num_items_recommended == 0 or user_num_items_recommended == 1:
            return 1.0
        else:
            return self.users_local_ild[uid] / (
                user_num_items_recommended * (user_num_items_recommended - 1) / 2
            )

    def update_recommendation(self, uid: int, item: int, reward: float):
        """update_recommendation.

        Args:
            uid (int): user id
            item (int): item id
            reward (float): reward
        """
        self.users_local_ild[uid] += np.sum(
            self.items_distance[self.users_items_recommended[uid], item]
        )
        self.users_items_recommended[uid].append(item)
