from tokenize import Number
from typing import List
import numpy as np


class Dataset:

    num_users = 0
    num_items = 0
    rate_domain = set()
    max_uid = 0
    max_iid = 0
    mean_rating = 0
    min_rating = 0
    max_rating = 0

    def __init__(
            self,
            data: np.ndarray
    ):
        """__init__

        Args:
            data (np.ndarray): the data

        """
        self.data = data
        self.num_total_users = 0
        self.num_total_items = 0

    @staticmethod
    def normalize_ids(ids: List) -> np.array:
        """normalize_ids

            normalizes the ids by putting them in sequence

        Args:
            ids (List): list of ids

        Returns:
            result (np.array): the normalized ids
        """
        unique_values = np.sort(np.unique(ids))
        result = np.searchsorted(unique_values, ids)
        return result

    def reset_index(self):
        """reset_index

            Resets user and item indices

        """
        self.data[:, 0] = self.normalize_ids(self.data[:, 0])
        self.data[:, 1] = self.normalize_ids(self.data[:, 1])

    def set_parameters(self):

        """set_parameters

           Calculates and updates the database parameters

        """   
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

    def update_num_total_users_items(self, num_total_users=0, num_total_items=0):
        """update_num_total_users_items

           Updates the total number of users and items

        """ 
        self.num_total_users = num_total_users if num_total_users > self.max_uid+1 else self.max_uid+1
        self.num_total_items = num_total_items if num_total_items > self.max_iid+1 else self.max_iid+1


    def __str__(self):
        return f"\n\
                 Number of users: {self.num_users} \n\
                 Number of itemns: {self.num_items} \n\
                 Max uid: {self.max_uid} \n\
                 Max iid: {self.max_iid} \n\
                 Min rating: {self.min_rating} \n\
                 Max rating: {self.max_rating} \n\
                 Total number of users: {self.num_total_users} \n\
                 Total number of items: {self.num_total_items} \n"