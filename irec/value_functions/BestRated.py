import numpy as np
from tqdm import tqdm
from .ExperimentalValueFunction import ExperimentalValueFunction
from .BestRated import *
import matplotlib.pyplot as plt
import os
import scipy
from collections import defaultdict
import random


class BestRated(ExperimentalValueFunction):
    """Best Rated.
    
    Recommends top-rated items based on their average ratings in each iteration.    
    """
    def __init__(self, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_bestrated(consumption_matrix):
        """get_items_bestrated.
        
        Args:
            consumption_matrix: consumption_matrix
            normalize (bool): normalize

        Returns:
            numpy.ndarray:
        """

        items_bestrated = np.array(np.mean(consumption_matrix, axis=0)).flatten()
        return items_bestrated

    def reset(self, observation):
        """reset.

        Args:
            observation: 
        """
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.num_total_items = self.train_dataset.num_total_items
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (self.train_dataset.data[:, 2],
             (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1])),
            (self.train_dataset.num_total_users,
             self.train_dataset.num_total_items))
        self.null_value = np.min(self.train_consumption_matrix)
        self.items_bestrated = self.get_items_bestrated(
            self.train_consumption_matrix)

    def action_estimates(self, candidate_actions):
        """action_estimates.

        Args:
            candidate_actions: (user id, candidate_items)
        
        Returns:
            numpy.ndarray:
        """
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = self.items_bestrated[candidate_items]

        return items_score, None

    def update(self, observation, action, reward, info):
        """update.

        Args:
            observation:
            action: (user id, item)
            reward (float): reward
            info: 
        """
        uid = action[0]
        item = action[1]
        additional_data = info
        if reward != self.null_value:
            self.items_bestrated[item] = self.train_consumption_matrix[:,item].toarray().mean()
