import numpy as np
from tqdm import tqdm
from . import Interactor
import os
import random
import scipy.stats
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

        for i in tqdm(range(self.get_iterations())):
            for idx_uid in range(num_users):
                uid = uids[idx_uid]
                # best_item = np.argmax([self.beta_distributions[item].rvs() for item in range(num_items)])
                not_recommended = np.ones(num_items,dtype=bool)
                not_recommended[self.result[uid]] = 0
                items_not_recommended = np.nonzero(not_recommended)[0]
                best_item = items_not_recommended[np.argmax(np.random.beta(self.alphas[items_not_recommended],
                                                     self.betas[items_not_recommended]))]
                # reward = (self.get_reward(uid,best_item)-self.lowest_value)/(self.highest_value-self.lowest_value)
                # reward = 1 if reward >= 0.8 else 0
                reward = self.get_reward(uid,best_item)
                reward = 1 if reward >= self.values[-2] else 0
                
                self.alphas[best_item] += reward
                self.betas[best_item] += 1-reward
                # self.betas[best_item] += self.highest_value-reward
                self.result[uid].append(best_item)
        self.save_result()
