import numpy as np
from tqdm import tqdm
from .ExperimentalValueFunction import ExperimentalValueFunction
import matplotlib.pyplot as plt
import os
import scipy
from collections import defaultdict
import random
import itertools
from numba import jit, prange
import mf
from .MFValueFunction import MFValueFunction


class COFIBA(MFValueFunction):
    """COFIBA.
    
    This method relies on upper-confidence-based tradeoffs between exploration and exploitation,
    combined with adaptive clustering procedures at both the user and the item sides [1]_.

    References
    ----------
    .. [1] Li, Shuai, Alexandros Karatzoglou, and Claudio Gentile. "Collaborative filtering bandits." 
       Proceedings of the 39th International ACM SIGIR conference on Research and Development 
       in Information Retrieval. 2016.   
    """
    def __init__(self, alpha=1, alpha_2=1, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
            alpha:
            alpha_2:
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.alpha_2 = alpha_2


    def cb(self, alpha, item_latent_factors, m, t):
        """cb.

        Args:
            alpha: 
            item_latent_factors:
            m: 
            t: 

        Returns:
            float
        """
        return alpha * np.sqrt(item_latent_factors.T @ np.linalg.inv(m)
                               @ item_latent_factors * np.log10(t + 1))
        pass

    def update_user_cluster(self, uid, item):
        """update_user_cluster.

        Args:
            uid (int): user id
            item (int): item id

        Returns:
            :
        """
        item_cluster = self.items_clustering[item]
        users_graph = self.users_graphs[item_cluster].copy()
        neighbors = np.nonzero(users_graph[uid])[1]
        for neighbor in neighbors:
            if np.abs(self.users_latent_factors[uid] @ self.items_latent_factors[item] - self.users_latent_factors[neighbor] @ self.items_latent_factors[item])\
               > self.cb(self.alpha_2,self.items_latent_factors[item],self.users_m[uid], self.t) + self.cb(self.alpha_2,self.items_latent_factors[item],self.users_m[neighbor], self.t):
                users_graph[uid, neighbor] = 0
                users_graph[neighbor, uid] = 0
        n_components, labels = scipy.sparse.csgraph.connected_components(
            users_graph)
        # self.users_clusterings[item_cluster] = labels
        return users_graph, labels

    def update_item_cluster(self, uid, item):
        """update_item_cluster.

        Args:
            uid (int): user id
            item (int): item id

        """
        item_cluster = self.items_clustering[item]
        actual_cluster_items = set(
            np.nonzero(self.items_clustering == item_cluster)[0])

        neighbors = np.nonzero(self.items_graph[item])[1]
        # users_graph = self.users_graphs[item_cluster].copy()
        # neighbors = np.nonzero(users_graph[uid])[1]

        generated_user_neighbors = []
        for neighbor in neighbors:
            generated_user_neighbors = set()
            for uid2 in range(self.num_total_users):
                if np.abs(self.users_latent_factors[uid] @ self.items_latent_factors[neighbor] - self.users_latent_factors[uid2] @ self.items_latent_factors[neighbor])\
                <= self.cb(self.alpha_2,self.items_latent_factors[neighbor],self.users_m[uid], self.t) + self.cb(self.alpha_2,self.items_latent_factors[neighbor],self.users_m[uid2], self.t):
                    generated_user_neighbors.add(uid)
            if generated_user_neighbors != actual_cluster_items:
                self.items_graph[item, neighbor] = 0
                self.items_graph[neighbor, item] = 0

        n_components, labels = scipy.sparse.csgraph.connected_components(
            self.items_graph)
        self.items_clustering = labels
        r = range(self.items_n_components, n_components)
        self.items_n_components = n_components
        for i in r:
            users_graph = self.new_graph(self.num_total_users)
            self.users_graphs.append(users_graph)
            n_components, labels = scipy.sparse.csgraph.connected_components(
                users_graph)
            self.users_clusterings.append(labels)

    @staticmethod
    def new_graph(n):
        """new_graph.

        Args:
            n (int):

        Returns:
            sparse matrix:
        """
        graph = scipy.sparse.random(n,
                                    n,
                                    density=2 * np.log(n) / n,
                                    dtype=bool,
                                    format='lil')
        COFIBA.symmetrize_matrix(graph)
        for i in range(graph.shape[0]):
            graph[i, i] = 0
        graph.tocsr().eliminate_zeros()
        return graph

    @staticmethod
    def symmetrize_matrix(m):
        """symmetrize_matrix.

        Args:
            m (int):

        """
        for i in range(m.shape[0]):
            for j in range(m.shape[0]):
                if j < i:
                    m[j, i] = m[i, j]

    def score(self, uid, item, user_connected_component):
        """score.

        Args:
            uid (int): user id
            item (int): item id
            user_connected_component:

        Returns:
            :
        """
        neighbors = user_connected_component
        num_neighbors = len(neighbors)
        cluster_m = self.I + np.add.reduce(self.users_m[np.append(
            neighbors, uid)]) - num_neighbors * self.I
        cluster_b = np.add.reduce(self.users_b[np.append(neighbors, uid)])
        cluster_latent_factors = cluster_m @ cluster_b
        return cluster_latent_factors @ self.items_latent_factors[
            item] + self.cb(self.alpha, self.items_latent_factors[item],
                            cluster_m, self.t)

    def reset(self, observation):
        """reset.

        Args:
            observation: 
        """
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (self.train_dataset.data[:, 2],
             (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1])),
            (self.train_dataset.num_total_users,
             self.train_dataset.num_total_items))
        self.num_total_items = self.train_dataset.num_total_items
        self.consumption_matrix = self.train_consumption_matrix.tolil()
        self.num_total_users = self.train_dataset.num_total_users

        mf_model = mf.SVD(num_lat=self.num_lat)
        mf_model.fit(self.train_consumption_matrix)
        self.items_latent_factors = mf_model.items_weights

        self.I = np.identity(self.num_lat)

        self.items_graph = self.new_graph(self.num_total_items)

        self.items_n_components, self.items_clustering = scipy.sparse.csgraph.connected_components(
            self.items_graph)
        self.users_graphs = []
        self.users_clusterings = []
        for i in range(self.items_n_components):
            users_graph = self.new_graph(self.num_total_users)
            self.users_graphs.append(users_graph)
            n_components, labels = scipy.sparse.csgraph.connected_components(
                users_graph)
            self.users_clusterings.append(labels)
        self.users_b = np.zeros((self.num_total_users, self.num_lat))
        self.users_m = []
        for i in range(self.num_total_users):
            self.users_m.append(np.identity(self.num_lat))

        self.users_m = np.array(self.users_m)
        self.users_latent_factors = [
            np.linalg.inv(m) @ b for b, m in zip(self.users_b, self.users_m)
        ]
        self.t = 1
        self.recent_predict = True

    def action_estimates(self, candidate_actions):
        """action_estimates.

        Args:
            candidate_actions: (user id, candidate_items)

        Returns:
            numpy.ndarray:
        """
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = np.zeros(candidate_items.shape)
        for i, item in enumerate(candidate_items):
            users_graph, labels = self.update_user_cluster(uid, item)
            user_connected_component = np.nonzero(labels[uid] == labels)[0]
            items_score[i] = self.score(uid, item, user_connected_component)
        self.recent_predict = True
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
        users_graph, labels = self.update_user_cluster(uid, item)
        item_cluster = self.items_clustering[item]
        self.users_graphs[item_cluster] = users_graph
        self.update_item_cluster(uid, item)
        if self.recent_predict:
            self.t += 1
            self.recent_predict = False
