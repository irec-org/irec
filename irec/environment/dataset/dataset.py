import numpy as np


class Dataset:

    def __init__(
        self,
        data,
        num_total_users=None,
        num_total_items=None,
        num_users=None,
        num_items=None,
        rate_domain=None,
        uids=None,
    ):
        self.data = data
        self.num_users = num_users
        self.num_items = num_items
        self.rate_domain = rate_domain
        self.uids = uids
        self.num_total_users = num_total_users
        self.num_total_items = num_total_items

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