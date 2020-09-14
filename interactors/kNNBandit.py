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

    def interact(self):
        super().interact()
        uids = self.test_users
        num_items = self.train_consumption_matrix.shape[1]
        consumption_matrix = self.train_consumption_matrix.tolil()
        total_num_users = self.train_consumption_matrix.shape[0]
        # items_popularity = MostPopular.get_items_popularity(self.train_consumption_matrix, normalize=False)

        # top_iids = list(reversed(np.argsort(items_popularity)))[:self.get_iterations()]
        num_users = len(uids)
        users_num_interactions = defaultdict(int)
        available_users = set(uids)

        # users_alpha = defaultdict(int)
        # users_alpha = np.zeros(self.train_consumption_matrix.shape[0],self.train_consumption_matrix.shape[0])
        users_alphas = (self.train_consumption_matrix @ self.train_consumption_matrix.T).A
        users_rating_sum = self.train_consumption_matrix.sum(axis=1).A.flatten()
        # for uid in range(self.train_consumption_matrix.shape[0]):
        #     users_alpha
        # self.train_consumption_matrix[uid1]
        # i=0
        # for uid1, uid2 in itertools.combinations(range(self.train_consumption_matrix.shape[0]),2):
        #     users_alpha[uid1, uid2] = np.dot(self.train_consumption_matrix[uid1], self.train_consumption_matrix[uid2].A)
        #     print(i)
        #     i+=1
        for i in tqdm(range(num_users*self.interactions)):
            uid = random.sample(available_users,k=1)[0]
            not_recommended = np.ones(num_items,dtype=bool)
            not_recommended[self.results[uid]] = 0
            items_not_recommended = np.nonzero(not_recommended)[0]
            # print(consumption_matrix[uid].T)
            # ratings_similarity = consumption_matrix @ consumption_matrix[uid].A.flatten()
            # alphas = ratings_similarity
            # betas = consumption_matrix.sum(axis=1).A.flatten() - alphas

            # print(users_alphas.shape)
            # print(users_rating_sum.shape)
            # print(users_rating_sum-users_alphas[uid])
            # print(users_alphas[uid])
            users_score = np.zeros(total_num_users)
            for i, v1, v2 in zip(list(range(total_num_users)),users_alphas[uid], users_rating_sum-users_alphas[uid]):
                if v1 > 0 and v2 > 0:
                    users_score[i] = np.random.beta(v1,v2)
                else:
                    users_score[i] = 0

            # users_score = np.random.beta(users_alphas[uid], users_rating_sum-users_alphas[uid])
            for i in users_score.argsort():
                if i != uid:
                    top_user = uid
                    break
                
            top_items = np.argsort(consumption_matrix[top_user].A.flatten()[items_not_recommended])[::-1][:self.interaction_size]
            best_items = items_not_recommended[top_items]

            self.results[uid].extend(best_items)

            for item in best_items:
                tmp_val =self.get_reward(uid,item)*self.get_reward(top_user,item,from_test_and_train=True)
                users_alphas[uid,top_user] = tmp_val
                users_alphas[top_user,uid] = tmp_val
                users_rating_sum[uid] += self.get_reward(uid,item)
                consumption_matrix[uid, item] = self.get_reward(uid,item)
            
            users_num_interactions[uid] += 1
            if users_num_interactions[uid] == self.interactions:
                available_users = available_users - {uid}

        self.save_results()
