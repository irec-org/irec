import numpy as np
from tqdm import tqdm
from .ExperimentalInteractor import ExperimentalInteractor
import matplotlib.pyplot as plt
import os
import scipy.sparse
from collections import defaultdict
import random
import itertools


class kNNBandit(ExperimentalInteractor):

    def __init__(self, alpha_0, beta_0, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.k = k
        self.parameters.extend(['alpha_0', 'beta_0','k'])

    def train(self, train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (self.train_dataset.data[:, 2],
             (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1])),
            (self.train_dataset.num_total_users,
             self.train_dataset.num_total_items))
        self.train_consumption_matrix = (self.train_consumption_matrix>=4).astype(int)

        self.num_total_users = self.train_dataset.num_total_users
        self.num_total_items = self.train_dataset.num_total_items
        self.consumption_matrix = self.train_consumption_matrix.tolil()

        self.users_alphas = (
            self.train_consumption_matrix @ self.train_consumption_matrix.T).A
        self.users_rating_sum = self.train_consumption_matrix.sum(
            axis=1).A.flatten() + self.alpha_0 + self.beta_0
        
        del self.train_consumption_matrix

    def predict(self, uid, candidate_items, num_req_items):
        users_score = np.zeros(self.num_total_users)
        uids = np.array(list(set(range(self.num_total_users)) - {uid}))
        vs1 = self.users_alphas[uid,uids]
        vs2 = self.users_rating_sum[uids] - self.users_alphas[uid,uids]
        for i, uid_ in enumerate(uids):
            v1 = vs1[i]
            v2 = vs2[i]
            if v1 > 0 and v2 > 0:
                users_score[uid_] = np.random.beta(v1, v2)

        top_uids = np.argpartition(users_score,-self.k)[-self.k:]

        items_score = (users_score[top_uids].reshape(-1,1)*self.consumption_matrix[top_uids,:][:,candidate_items].A).sum(axis=0)

        return items_score, None

    def update(self, uid, item, reward, additional_data):
        u1_reward = reward>=4
        self.users_rating_sum[uid] += u1_reward
        self.consumption_matrix[uid, item] = u1_reward

        # u2_reward = self.consumption_matrix[uids, item].A.flatten()
        u2_reward = self.consumption_matrix[:,item].A.flatten()
        tmp_val = np.sum(u1_reward * u2_reward)
        self.users_alphas[uid, :] += tmp_val
        self.users_alphas[:, uid] += tmp_val
