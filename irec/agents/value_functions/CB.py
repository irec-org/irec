"""
Cluster-Based algorithm of "Cluster-Based Bandits: Fast Cold-Start for Recommender System New Users"
"""
import numpy as np
from .ExperimentalValueFunction import ExperimentalValueFunction
import scipy.stats
from collections import defaultdict
from sklearn.cluster import KMeans
import itertools
import mf

from cachetools import cached
from cachetools.keys import hashkey


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
        self,
        num_clusters: int,
        B: float,
        C: float,
        D: float,
        num_lat: int,
        # cache_dir: str,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.B = B
        self.C = C
        self.D = D
        self.num_lat = num_lat
        self.num_clusters = num_clusters

        # self.cache_dir = cache_dir

        # self.Gamma = cached(self.Gamma,)

    # def find_object(db_handle, query):
    # print("processing {0}".format(query))
    # return query
    # memory = Memory(self.cache_dir, verbose=0)
    # self.Gamma =  memory.cache(self.Gamma, ignore=['self'])
    # # self.Rn =  memory.cache(self.Rn, ignore=['self'])
    # self.Sigma =  memory.cache(self.Sigma, ignore=['self'])
    # self.alpha =  memory.cache(self.alpha, ignore=['self'])
    @cached(cache={}, key=lambda self, g, h, v: hashkey(g, h, v))
    def Gamma(self, g, h, v):

        return (self.groups_mean[g][v] - self.groups_mean[h][v]) ** 2 / self.groups_std[
            g
        ][v]

    @cached(cache={}, key=lambda self, i, g, h: hashkey(i, g, h))
    def alpha(self, i, g, h):
        return self.Gamma(g, h, i) / np.sum(
            [self.Gamma(g, h, v) for v in range(self.num_total_items)]
        )

    @cached(cache={}, key=lambda self, g, h: hashkey(g, h))
    def Sigma(self, g, h):
        c = 0
        for v in range(self.num_total_items):
            c += self.Gamma(g, h, v)
        return c

    def explore(self, g, user_candidate_items):
        h = np.argmin([self.Sigma(g, h) for h in self.groups])
        return [self.Gamma(g, h, v) for v in user_candidate_items]

    def Rn(self, uid, g, h):
        # items = [
        # vi
        # for vi in range(self.num_total_items)
        # if self.groups_mean[g, vi] != self.groups_mean[h, vi]
        # ]
        items = np.arange(self.num_total_items)
        items = items[self.groups_mean[g, items] != self.groups_mean[h, items]]
        # self.[]
        alphas = np.array([self.alpha(vi, g, h) for vi in items])
        with np.errstate(invalid="ignore"):
            l = alphas * (
                (
                    self.consumption_matrix[uid, items].toarray().flatten()
                    - self.groups_mean[h, items].flatten()
                )
                / (
                    self.groups_mean[g, items].flatten()
                    - self.groups_mean[h, items].flatten()
                )
            )
        # print(np.sum(l))
        # l = [
        # self.alpha(vi, g, h)
        # * (
        # (self.consumption_matrix[uid, vi] - self.groups_mean[h, vi])
        # / (self.groups_mean[g, vi] - self.groups_mean[h, vi])
        # )
        # for vi in range(self.num_total_items)
        # if self.groups_mean[g, vi] != self.groups_mean[h, vi]
        # ]
        return np.sum(l)

    def I(self, uid, g):
        l = []
        for h in range(self.num_clusters):
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

        # self.train_consumption_matrix

        mf_model = mf.SVD(num_lat=self.num_lat)
        mf_model.fit(self.train_consumption_matrix)
        kmeans = KMeans(self.num_clusters).fit(mf_model.users_weights)
        self.num_total_items = self.train_dataset.num_total_items
        self.num_total_users = self.train_dataset.num_total_users

        self.groups_users = defaultdict(list)
        for uid, group in enumerate(kmeans.labels_):
            self.groups_users[group].append(uid)

        self.groups = list(range(self.num_clusters))
        self.groups_mean = np.zeros((self.num_clusters, self.num_total_items))
        self.groups_std = np.zeros((self.num_clusters, self.num_total_items))
        for group, uids in self.groups_users.items():
            ratings = self.train_consumption_matrix[uids]
            self.groups_mean[group] = ratings.mean(axis=0)
            num = np.array((ratings > 0).sum(axis=0)).flatten()
            # print(num.shape)
            try:
                self.groups_std[group] = _stds(ratings, axis=0)
            except:
                self.groups_std[group] = np.zeros(self.num_total_items)
            self.groups_std[group] += 0.5 * np.sqrt(np.log(1 / 0.2) / (num + 0.01))
            # print(group, self.groups_std[group])
        # self.groups_std[self.groups_std] +=

        self.consumption_matrix = self.train_consumption_matrix.todok()
        self.exploration_phase = defaultdict(lambda: True)
        self.new_user = defaultdict(lambda: True)
        self.users_group = {}

        # self.Gamma_cache = dict()
        # for g in range(self.num_clusters):
        # for h in range(self.num_clusters):
        # for v in range(self.num_total_items):
        # self.Gamma_cache[(g,h,v)] = self.Gamma(g,h,v)

        # self.Rn_cache = dict()
        # for g in range(self.num_clusters):
        # for h in range(self.num_clusters):
        # for u in range(self.num_total_users):
        # self.Rn_cache[(u,g,h)] = self.Rn(u,g,h)

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
            # print("Not new user")
            if self.exploration_phase[uid]:
                # [g for g in self.groups if self.Rn(uid,g,)]
                candidates_groups = []
                # print("Not new user c1")
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
                # print("Not new user c2")
                if len(candidates_groups) != 0:
                    g_hat = np.argmax([self.I(uid, g) for g in self.groups])
                    result_explore = self.explore(g_hat, candidate_items)
                    cond1 = (
                        np.min([self.Sigma(g_hat, h) for h in range(self.num_clusters)])
                        >= self.B
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
