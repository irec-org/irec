import numpy as np
import scipy.sparse
from collections import defaultdict
from .base import Metric
from typing import Any

np.seterr(all="raise")


class EPD(Metric):
    """Expected Profile Distance.

    EPD, on the other hand, is a distance-based novelty measure, which looks
    at distances between the items inthe user’s profile and the recommended items.
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
        self.users_consumed_items = defaultdict(list)
        self.users_items_recommended = defaultdict(list)
        self.users_local_ild = defaultdict(float)

        self.users_relevant_items = scipy.sparse.csr_matrix(
                (
                    self.ground_truth_dataset.data[:, 2],
                    (
                        self.ground_truth_dataset.data[:, 0].astype(int),
                        self.ground_truth_dataset.data[:, 1].astype(int),
                    ),
                ),
                shape=(self.ground_truth_dataset.num_total_users, self.ground_truth_dataset.num_total_items),
            )
        # print(self.users_relevant_items)
        # print("min rating",self.ground_truth_dataset.min_rating)
        self.users_relevant_items[
            self.users_relevant_items >= self.ground_truth_dataset.min_rating
        ] = True
        # print(self.users_relevant_items)

    def compute(self, uid: int):
        """compute.

        Args:
            uid (int): user id
        """
        rel = np.array(self.users_relevant_items[uid].A).flatten()
        consumed_items = self.users_consumed_items[uid]
        predicted = self.users_items_recommended[uid]
        res = (
            rel[predicted][:, None]
            @ rel[consumed_items][None, :]
            * self.items_distance[predicted, :][:, consumed_items]
        )
        print(">>>>>>>>>", len(predicted), consumed_items, self.users_consumed_items)
        C = 1 / (len(predicted) * np.sum(rel[consumed_items]))
        return C * np.sum(res)

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

    def update_consumption_history(self, uid: int, item: int, reward: float):
        """update_consumption_history.

        Args:
            uid (int): user id
            item (int): item id
            reward (float): reward
        """
        print("ENTROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOU\n\n\n\n")
        self.users_consumed_items[uid].append(item)
