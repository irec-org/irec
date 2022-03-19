from typing import List

import numpy as np


class Dataset:

    def __init__(
            self,
            data
    ):
        self.data = data

    @staticmethod
    def _normalize_ids(ids: List) -> np.array:
        unique_values = np.sort(np.unique(ids))
        result = np.searchsorted(unique_values, ids)
        return result

    def reset_index(self) -> np.array:
        self.data[:, 0] = self._normalize_ids(self.data[:, 0])
        self.data[:, 1] = self._normalize_ids(self.data[:, 1])

    def set_parameters(self):
        self.num_users = len(np.unique(self.data[:, 0]))
        self.num_items = len(np.unique(self.data[:, 1]))
        self.rate_domain = set(np.unique(self.data[:, 2]))
        self.uids = np.unique(self.data[:, 0]).astype(int)
        self.iids = np.unique(self.data[:, 1]).astype(int)
        self.max_uid = np.max(self.uids)
        self.max_iid = np.max(self.iids)
        self.mean_rating = np.mean(self.data[:, 2])
        self.min_rating = np.min(self.data[:, 2])
        self.max_rating = np.max(self.data[:, 2])
        self.num_total_users = self.max_uid + 1
        self.num_total_items = self.max_iid + 1

    def load(self):
        return
