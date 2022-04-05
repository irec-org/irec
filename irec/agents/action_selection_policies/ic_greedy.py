from ..value_functions.most_popular import MostPopular
from ..value_functions.log_pop_ent import LogPopEnt
from ..value_functions.entropy import Entropy
from .base import ActionSelectionPolicy
from collections import defaultdict
from sklearn.cluster import KMeans
from irec.mf.SVD import SVD
from typing import Any
import scipy.sparse
import numpy as np
import random

class ASPICGreedy(ActionSelectionPolicy):
    def __init__(self, stop, num_clusters, num_lat, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = stop
        self.num_clusters = num_clusters
        self.num_lat = num_lat

    def choose_group(self, uid):
        b = self.users_num_consumption_groups[uid]
        return np.random.choice(np.flatnonzero(b == b.min()))

    def get_item_group(self, item):
        return self.groups_items[item]

    def select_actions(self, actions, action_estimates, actions_num):
        uid = actions[0]
        if self.users_num_consumption[uid] >= self.stop:
            return (
                actions[0],
                actions[1][
                    np.argpartition(action_estimates, -actions_num)[-actions_num:]
                ],
            ), None
        else:
            i = 0
            while True:
                i += 1
                # print(i)
                # g = self.choose_group(uid)
                # try:
                g = random.randint(0, self.num_clusters)
                g_items = np.array(self.groups_items[g])
                # print(actions[1],g_items)
                g_items = np.intersect1d(actions[1], g_items)
                # print(g_items)
                if len(g_items) == 0:
                    continue
                g_items = list(g_items)
                top_item = g_items[np.argmax(self.items_popularity[g_items])]
                break
                # break
                # except:
                # continue

            return (
                actions[0],
                [top_item],
            ), None

    def update(self, observation, action, reward, info):
        uid = action[0]
        # iid = action[1]
        self.users_num_consumption[uid] += 1
        pass

    def reset(self, observation):
        train_dataset = observation
        self.train_dataset = train_dataset

        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (
                self.train_dataset.data[:, 2],
                (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1]),
            ),
            (self.train_dataset.num_total_users, self.train_dataset.num_total_items),
        )
        mf_model = SVD(num_lat=self.num_lat)
        mf_model.fit(self.train_consumption_matrix)
        self.items_weights = mf_model.items_weights
        self.num_latent_factors = len(self.items_weights[0])

        kmeans = KMeans(self.num_clusters).fit(self.items_weights)

        self.num_total_items = self.train_dataset.num_total_items
        self.num_total_users = self.train_dataset.num_total_users

        self.groups_items = defaultdict(list)
        print(len(kmeans.labels_))
        for uid, group in enumerate(kmeans.labels_):
            self.groups_items[group].append(uid)

        self.groups = list(range(self.num_clusters))
        self.users_num_consumption = defaultdict(int)
        self.users_num_consumption_groups: Any = defaultdict(
            lambda: np.zeros(self.num_clusters)
        )

        self.items_entropy = Entropy.get_items_entropy(self.train_consumption_matrix)
        self.items_popularity = MostPopular.get_items_popularity(
            self.train_consumption_matrix, normalize=False
        )
        self.items_logpopent = LogPopEnt.get_items_logpopent(
            self.items_popularity, self.items_entropy
        )