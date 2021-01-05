import numpy as np
from tqdm import tqdm
from .Interactor import Interactor
import matplotlib.pyplot as plt
import os
import scipy.sparse
from collections import defaultdict
import random
import itertools
class kNNBandit(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[2],(self.train_dataset.data[0],self.train_dataset.data[1])),(self.train_dataset.users_num,self.train_dataset.items_num))
        self.num_items = self.train_dataset.num_items

        self.consumption_matrix = self.train_consumption_matrix.tolil()
        self.total_num_users = self.train_consumption_matrix.shape[0]

        self.users_alphas = (self.train_consumption_matrix @ self.train_consumption_matrix.T).A
        self.users_rating_sum = self.train_consumption_matrix.sum(axis=1).A.flatten()

    def predict(self,uid,candidate_items):
        users_score = np.zeros(self.total_num_users)
        for i, v1, v2 in zip(list(range(self.total_num_users)),self.users_alphas[uid], self.users_rating_sum-self.users_alphas[uid]):
            if v1 > 0 and v2 > 0:
                users_score[i] = np.random.beta(v1,v2)
            else:
                users_score[i] = 0

        for i in users_score.argsort():
            if i != uid:
                top_user = uid

        return consumption_matrix[top_user].A.flatten()[candidate_items], {'top_user': top_user}
        # best_items = items_not_recommended[top_items]


    def update(self,uid,item,reward,additional_data):
        top_user = additional_data['top_user']
        u1_reward = reward
        u2_reward = self.consumption_matrix[top_user,item]
        tmp_val = u1_reward*u2_reward
        self.users_alphas[uid,top_user] = tmp_val
        self.users_alphas[top_user,uid] = tmp_val
        self.users_rating_sum[uid] += u1_reward
        self.consumption_matrix[uid, item] = u1_reward
        
