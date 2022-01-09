from typing import Any
import numpy as np
from tqdm import tqdm
from .ExperimentalValueFunction import ExperimentalValueFunction
import matplotlib.pyplot as plt
import os
import scipy.sparse
from collections import defaultdict
import random
import itertools
import random


class kNNBandit(ExperimentalValueFunction):
    """k Nearest-Neighbors Bandit.

    A simple multi-armed bandit elaboration of neighbor-based collaborative filtering.
    A variant of the nearest-neighbors scheme, but endowed with a controlled stochastic exploration
    capability of the users’ neighborhood, by a parameter-free application of Thompson sampling [1]_.

    References
    ----------
    .. [1] Sanz-Cruzado, Javier, Pablo Castells, and Esther López. "A simple multi-armed nearest-neighbor
       bandit for interactive recommendation." Proceedings of the 13th ACM Conference on Recommender Systems. 2019.
    """

    def __init__(self, alpha_0, beta_0, k, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
        """
        super().__init__(*args, **kwargs)
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.k = k

    def reset(self, observation):
        """reset.

        Args:
            observation:
        """
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csc_matrix(
            (
                self.train_dataset.data[:, 2],
                (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1]),
            ),
            (self.train_dataset.num_total_users, self.train_dataset.num_total_items),
        )
        self.train_consumption_matrix = (self.train_consumption_matrix >= 4).astype(int)

        self.num_total_users = self.train_dataset.num_total_users
        self.num_total_items = self.train_dataset.num_total_items
        self.consumption_matrix = self.train_consumption_matrix.tolil()

        self.users_alphas = np.zeros(
            (self.num_total_users, self.num_total_users), dtype="float32"
        )
        self.users_rating_sum = (
            np.zeros((self.num_total_users)) + self.alpha_0 + self.beta_0
        )
        l_ = range(len(self.train_dataset.data))
        self.items_consumed_users: Any = defaultdict(list)
        self.items_consumed_users_ratings: Any = defaultdict(list)
        for i in tqdm(l_):
            uid = int(self.train_dataset.data[i, 0])
            item = int(self.train_dataset.data[i, 1])
            reward = self.train_dataset.data[i, 2]
            reward = reward >= 4
            self.users_rating_sum[uid] += reward
            if len(self.items_consumed_users[item]) > 0:
                item_consumed_uids = np.array(
                    [i for i in self.items_consumed_users[item]]
                )
                item_ratings = np.array(
                    [i for i in self.items_consumed_users_ratings[item]]
                )
                self.users_alphas[uid, item_consumed_uids] += reward * item_ratings
            self.items_consumed_users[item].append(uid)
            self.items_consumed_users_ratings[item].append(reward)

        del self.train_consumption_matrix

    def action_estimates(self, candidate_actions):
        """action_estimates.

        Args:
            candidate_actions: (user id, candidate_items)

        Returns:
            numpy.ndarray:
        """
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        users_score = np.zeros(self.num_total_users - 1)
        uids = np.array(list(set(range(self.num_total_users)) - {uid}))
        vs1 = self.users_alphas[uid, uids]
        vs2 = self.users_rating_sum[uids] - self.users_alphas[uid, uids]
        for i, uid_ in enumerate(uids):
            v1 = vs1[i]
            v2 = vs2[i]
            users_score[i] = np.random.beta(v1 + self.alpha_0, v2 + self.beta_0)

        idxs = np.argpartition(users_score, -self.k)[-self.k :]
        top_uids = uids[idxs]

        top_users_score = users_score[idxs]

        items_score = np.zeros(len(candidate_items))
        for top_user_score, top_uid in zip(top_users_score, top_uids):
            items_score += (
                top_user_score
                * self.consumption_matrix[top_uid].A.flatten()[candidate_items]
            )

        idxs = np.where(np.max(items_score) == items_score)[0]
        items_score[idxs] += np.random.rand(len(idxs))

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
        reward = reward >= 4

        if len(self.items_consumed_users[item]) > 0:
            item_consumed_uids = np.array([i for i in self.items_consumed_users[item]])
            item_ratings = np.array(
                [i for i in self.items_consumed_users_ratings[item]]
            )
            self.users_alphas[uid, item_consumed_uids] += reward * item_ratings

        self.users_rating_sum[uid] += reward
        self.consumption_matrix[uid, item] = reward
        self.items_consumed_users[item].append(uid)
        self.items_consumed_users_ratings[item].append(reward)
