"""
Cluster-Based algorithm of "Cluster-Based Bandits: Fast Cold-Start for Recommender System New Users"
"""
import numpy as np
from tensorflow.python.eager.context import num_gpus
from tqdm import tqdm
from .ValueFunction import ValueFunction
from .ExperimentalValueFunction import ExperimentalValueFunction
import os
import random
import scipy.stats
from collections import defaultdict
from sklearn.cluster import KMeans
import itertools


def _vars(a, axis=None):
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))


def _stds(a, axis=None):
    # print(_vars(a, axis))
    return np.sqrt(_vars(a, axis))


def _argmin(d):
    if not d:
        return None
    min_val = min(d.values())
    return [k for k in d if d[k] == min_val][0]


class CB(ExperimentalValueFunction):
    def __init__(
        self, num_clusters: int, B: float, C: float, D: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.B = B
        self.C = C
        self.D = D
        self.num_clusters = num_clusters

    def Gamma(self, g, h, v):

        return (self.groups_mean[g][v] - self.groups_mean[h][v]) ** 2 / self.groups_std[
            g
        ][v]

    def alpha(self, i, g, h):
        return self.Gamma(g, h, i) / np.sum(
            [self.Gamma(g, h, v) for v in range(self.num_total_items)]
        )

    def Sigma(self, g, h):
        c = 0
        for v in range(self.num_total_items):
            c += self.Gamma(g, h, v)
        return c

    def explore(self, g, user_candidate_items):
        h = np.argmin([self.Sigma(g, h) for h in self.groups])
        return [self.Gamma(g, h, v) for v in user_candidate_items]

    def Rn(self, uid, g, h):
        l = [
            self.alpha(vi, g, h)
            * (
                (self.consumption_matrix[uid, vi] - self.groups_mean[h, vi])
                / (self.groups_mean[g, vi] - self.groups_mean[h, vi])
            )
            for vi in range(self.num_total_items) if self.groups_mean[g, vi] != self.groups_mean[h, vi]
        ]
        return np.sum(l)

    def I(self, uid, g):
        l = []
        for h in range(self.num_total_items):
            if g != h:
                l.append(self.Rn(uid, g, h))
        return np.min(l)

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset

        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (
                self.train_dataset.data[:, 2],
                (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1]),
            ),
            (self.train_dataset.num_total_users, self.train_dataset.num_total_items),
        )
        kmeans = KMeans(self.num_clusters).fit(self.train_consumption_matrix)
        self.num_total_items = self.train_dataset.num_total_items

        self.groups_users = defaultdict(list)
        for uid, group in enumerate(kmeans.labels_):
            self.groups_users[group].append(uid)

        self.groups = list(range(self.num_clusters))
        self.groups_mean = np.zeros((self.num_clusters, self.num_total_items))
        self.groups_std = np.zeros((self.num_clusters, self.num_total_items))
        for group, uids in self.groups_users.items():
            ratings = self.train_consumption_matrix[uids]
            self.groups_mean[group] = ratings.mean(axis=0)
            self.groups_std[group] = _stds(ratings, axis=0)
            print(group,self.groups_std[group])
        self.groups_std[self.groups_std<0.01] = 0.01

        self.consumption_matrix = self.train_consumption_matrix.todok()
        self.exploration_phase = defaultdict(lambda: True)
        self.new_user = defaultdict(lambda: True)
        self.users_group = {}
        # print(self.groups)

        # for uid in range(self.train_dataset.data.shape[0]):
        # uid = int(self.train_dataset.data[uid, 0])
        # item = int(self.train_dataset.data[uid, 1])
        # reward = self.train_dataset.data[uid, 2]
        # # self.update(uid,item,reward,None)
        # self.update(None, (uid, item), reward, None)

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        if self.new_user[uid]:
            g = np.random.randint(self.num_clusters)
            return self.explore(g, candidate_items), None
        else:
            if self.exploration_phase[uid]:
                # [g for g in self.groups if self.Rn(uid,g,)]
                candidates_groups = []
                for g in self.groups:
                    ngroups = set(self.groups) - {g}
                    hs_vals = []
                    for h in ngroups:
                        hs_vals.append(self.Rn(uid, g, h))
                    hs_vals = np.abs(hs_vals)
                    min_h = np.min(hs_vals)
                    fval = np.abs(min_h - 1)
                    if fval <= self.C:
                        candidates_groups.append(g)
                if len(candidates_groups) != 0:
                    g_hat = np.argmax([self.I(uid, g) for g in self.groups])
                    result_explore = self.explore(g_hat, candidate_items)
                    cond1 = np.sum(
                        [
                            self.groups_mean[g_hat, vi] ** 2
                            / self.groups_std[g_hat, vi]
                            for vi in range(self.num_total_items)
                        ]
                    )
                    if cond1 or (
                        len(candidates_groups) == 1
                        and self.num_total_items > self.D * np.log2(self.num_clusters)
                    ):
                        self.users_group[uid] = g_hat
                        self.exploration_phase[uid] = False
                    return result_explore, None
                else:
                    sigma_values = dict()
                    for g, h in itertools.permutations(self.groups, 2):
                        sigma_values[(g, h)] = self.Sigma(g, h)
                    g, h = _argmin(sigma_values)
                    return self.explore(g, candidate_items), None
            else:
                user_g = self.users_group[uid]

                return [self.groups_mean[user_g][i] for i in candidate_items], None

        # return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        self.consumption_matrix[uid, item] = reward
        self.new_user[uid] = False
        # self.items_count[item] += 1
