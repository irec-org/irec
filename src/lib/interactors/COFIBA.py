import numpy as np
from tqdm import tqdm
from .ExperimentalInteractor import ExperimentalInteractor
import matplotlib.pyplot as plt
import os
import scipy.sparse
from collections import defaultdict
import random
import itertools
from numba import jit, prange
class COFIBA(ExperimentalInteractor):
    def __init__(self, alpha=1, alpha_2=1,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.alpha_2 = alpha_2
        self.parameters.extend(['alpha','alpha_2'])

    def cb(self, alpha, item_latent_factors, m, t):
        return alpha*np.sqrt(item_latent_factors.T @ np.linalg.inv(m) @ item_latent_factors * np.log10(t+1))
        pass
    def update_user_cluster(self, uid, item):
        item_cluster = self.items_clustering[item]
        users_graph = self.users_graphs[item_cluster].copy()
        neighbors = np.nonzero(users_graph[uid])[1]
        for neighbor in neighbors:
            if np.abs(self.users_latent_factors[uid] @ self.items_latent_factors[item] - self.users_latent_factors[neighbor] @ self.items_latent_factors[item])\
               > self.cb(self.alpha_2,self.items_latent_factors[item],self.users_m[uid], self.t) + self.cb(self.alpha_2,self.items_latent_factors[item],self.users_m[neighbor], self.t):
                users_graph[uid,neighbor] = 0
                users_graph[neighbor,uid] = 0
        n_components, labels = scipy.sparse.csgraph.connected_components(users_graph)
        # self.users_clusterings[item_cluster] = labels
        return users_graph, labels
    def update_item_cluster(self, uid, item):
        item_cluster = self.items_clustering[item]
        actual_cluster_items = set(np.nonzero(self.items_clustering == item_cluster)[0])
        
        neighbors = np.nonzero(self.items_graph[item])[1]
        # users_graph = self.users_graphs[item_cluster].copy()
        # neighbors = np.nonzero(users_graph[uid])[1]

        generated_user_neighbors = []
        for neighbor in neighbors:
            generated_user_neighbors = {}
            for uid2 in range(self.total_num_users):
                if np.abs(self.users_latent_factors[uid] @ self.items_latent_factors[neighbor] - self.users_latent_factors[uid2] @ self.items_latent_factors[neighbor])\
                <= self.cb(self.alpha_2,self.items_latent_factors[neighbor],self.users_m[uid], self.t) + self.cb(self.alpha_2,self.items_latent_factors[neighbor],self.users_m[uid2], self.t):
                    generated_user_neighbors.add(uid)
            if generated_user_neighbors != actual_cluster_items:
                self.items_graph[item, neighbor] = 0
                self.items_graph[neighbor, item] = 0

                
        n_components, labels = scipy.sparse.csgraph.connected_components(self.items_graph)
        self.items_clustering = labels
        r = range(self.items_n_components,n_components)
        self.items_n_components = n_components
        for i in r:
            users_graph = self.new_graph(total_num_users)
            self.users_graphs.append(users_graph)
            n_components, labels = scipy.sparse.csgraph.connected_components(users_graph)
            self.users_clusterings.append(labels)

    @staticmethod
    def new_graph(n):
        graph = scipy.sparse.random(n, n, density=3*np.log(n)/n, dtype=bool).tolil()
        COFIBA.symmetrize_matrix(graph)
        for i in range(graph.shape[0]):
            graph[i,i] = 0
        graph.tocsr().eliminate_zeros()
        return graph

    @staticmethod
    def symmetrize_matrix(m):
        for i in range(m.shape[0]):
            for j in range(m.shape[0]):
                if j < i:
                    m[j,i] = m[i,j]

    def score(uid, item, user_connected_component):
        neighbors = user_connected_component
        num_neighbors = len(neighbors)
        cluster_m = I + np.sum(users_m[neighbors+uid]) - num_neighbors*I
        cluster_b = np.sum(users_b[neighbors+uid])
        cluster_latent_factors = cluster_m @ cluster_b
        return cluster_latent_factors @ self.items_latent_factors[item] + self.cb(self.alpha,self.items_latent_factors[item],cluster_m,self.t)

    # def train(self,train_dataset):
    #     super().train(train_dataset)
    #     self.train_dataset = train_dataset
    #     self.train_consumption_matrix = scipy.sparse.csr_matrix((train_data[:,2],(train_data[:,0],train_data[:,1])),(self.train_dataset.users_num,self.train_dataset.items_num))
    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[:,2],(self.train_dataset.data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.users_num,self.train_dataset.items_num))
        self.num_items = self.train_dataset.num_items
    # def train(self,train_data,total_num_users,total_num_items):
    #     super().train(train_data)
    #     self.train_consumption_matrix = scipy.sparse.csr_matrix((train_data[:,2],(train_data[:,0],train_data[:,1])))
    #     self.num_items = self.train_consumption_matrix.shape[1]
        self.consumption_matrix = self.train_consumption_matrix.tolil()
        self.total_num_users = self.train_consumption_matrix.shape[0]

        mf_model = mf.SVD()
        mf_model.fit(self.train_consumption_matrix)
        self.items_latent_factors = mf_model.items_weights
        self.num_latent_factors = len(self.items_latent_factors[0])

        self.I = np.identity(self.num_latent_factors)
        # code her
        # ...
        self.items_graph = self.new_graph(self.num_items)
        # for i in range(num_items):
        #     for j in range(num_items):
        #         if j < i:
        #             self.items_graph[j,i] = self.items_graph[i,j]
            
        self.items_n_components, self.items_clustering = scipy.sparse.csgraph.connected_components(self.items_graph)
        self.users_graphs = []
        self.users_clusterings = []
        for i in range(self.items_n_components):
            users_graph = self.new_graph(total_num_users)
            self.users_graphs.append(users_graph)
            n_components, labels = scipy.sparse.csgraph.connected_components(users_graph)
            self.users_clusterings.append(labels)
        self.users_b = np.zeros((total_num_users,self.num_latent_factors))
        self.users_m = []
        for i in range(total_num_users):
            self.users_m.append(np.identity(self.num_latent_factors))

        self.users_latent_factors = [np.linalg.inv(m) @ b for b, m in zip(self.users_b,self.users_m)]
        # users_m = np.zeros(total_num_users,num_latent_factors,num_latent_factors)
        # users_m[
        # print(items_graph)
        # print(items_graph.sum())
        # raise SystemExit
        # for i in tqdm(range(num_users*self.interactions)):
        #     uid = random.sample(available_users,k=1)[0]
        #     not_recommended = np.ones(self.num_items,dtype=bool)
        #     not_recommended[self.results[uid]] = 0
        #     items_not_recommended = np.nonzero(not_recommended)[0]

        #     # code her
        #     # ...

    def predict(self,uid,candidate_items,num_req_items):
        items_score = np.zeros(candidate_items.shape)
        for i, item in enumerate(candidate_items):
            users_graph, labels = self.update_user_cluster(uid,item)
            user_connected_component = np.nonzero(labels[item] == labels)[0]
            items_score[i] = self.score(uid,item,user_connected_component)
                # best_items = items_not_recommended[np.argsort(items_score)][::-1][self.interaction_size]
                # for item in best_items:
        return items_score, None


    def update(self,uid,item,reward,additional_data):
        users_graph, labels = self.update_user_cluster(uid,item)
        item_cluster = self.items_clustering[item]
        self.users_graphs[item_cluster] = users_graph

            # self.t += 1
            # self.results[uid].extend(best_items)

            # for item in best_items:
            #     u1_reward = self.get_reward(uid,item)
            #     u2_reward = self.get_reward(top_user,item,from_test_and_train=True)
            #     tmp_val = u1_reward*u2_reward
            #     users_alphas[uid,top_user] = tmp_val
            #     users_alphas[top_user,uid] = tmp_val
            #     users_rating_sum[uid] += u1_reward
            #     consumption_matrix[uid, item] = u1_reward
            
        #     users_num_interactions[uid] += 1
        #     if users_num_interactions[uid] == self.interactions:
        #         available_users = available_users - {uid}
        #     self.t += 1

        # self.save_results()
