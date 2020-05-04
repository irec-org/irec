import numpy as np
from tqdm import tqdm
from . import Interactor
import os
import random
import scipy.stats
from collections import defaultdict
class ThompsonSampling(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def interact(self, uids):
        super().interact()
        num_users = len(uids)
        num_items = self.consumption_matrix.shape[1]
        
        # self.beta_distributions = [scipy.stats.beta(a=1,b=1) for item in range(num_items)]
        # print(self.beta_distributions[0].a)
        self.alphas = np.ones(num_items)
        self.betas = np.ones(num_items)

        users_num_interactions = defaultdict(int)
        available_users = set(uids)

        mask = np.ones(self.consumption_matrix.shape[0], dtype=bool)
        mask[uids] = 0
        self.alphas += np.count_nonzero(self.consumption_matrix[mask]>=self.threshold,
                         axis=0)
        self.betas += np.count_nonzero(self.consumption_matrix[mask]<self.threshold,
                         axis=0)

        for i in tqdm(range(num_users*self.interactions)):
            uid = random.sample(available_users,k=1)[0]

            # best_item = np.argmax([self.beta_distributions[item].rvs() for item in range(num_items)])
            not_recommended = np.ones(num_items,dtype=bool)
            not_recommended[self.result[uid]] = 0
            items_not_recommended = np.nonzero(not_recommended)[0]
            items_score = np.random.beta(self.alphas[items_not_recommended],
                                         self.betas[items_not_recommended])
            top_items = list(reversed(np.argsort(items_score)))[:self.interaction_size]
            best_items = items_not_recommended[top_items]
            self.result[uid].extend(best_items)
            # reward = (self.get_reward(uid,best_item)-self.lowest_value)/(self.highest_value-self.lowest_value)
            # reward = 1 if reward >= 0.8 else 0

            user_num_interactions = users_num_interactions[uid]

            for best_item in self.result[uid][user_num_interactions*self.interaction_size:(user_num_interactions+1)*self.interaction_size]:
                reward = self.get_reward(uid,best_item)
                reward = 1 if reward >= self.threshold else 0
                self.alphas[best_item] += reward
                self.betas[best_item] += 1-reward

            users_num_interactions[uid] += 1
            if users_num_interactions[uid] == self.interactions:
                available_users = available_users - {uid}
        self.save_result()
