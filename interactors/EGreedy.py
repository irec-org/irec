import numpy as np
from tqdm import tqdm
from . import Interactor
import os
import random
import scipy.stats
from collections import defaultdict
class EGreedy(Interactor):
    def __init__(self, epsilon=0.1,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def interact(self, uids):
        super().interact()
        num_users = len(uids)
        num_items = self.consumption_matrix.shape[1]
        
        items_mean_values = np.zeros(num_items)
        items_count = np.zeros(num_items,dtype=int)

        users_num_interactions = defaultdict(int)
        available_users = set(uids)

        mask = np.ones(self.consumption_matrix.shape[0], dtype=bool)
        mask[uids] = 0
        items_mean_values = np.mean(self.consumption_matrix[mask],axis=0)
        items_count += self.consumption_matrix[mask].shape[0]

        for i in tqdm(range(num_users*self.interactions)):
            uid = random.sample(available_users,k=1)[0]

            for j in range(self.interaction_size):
                not_recommended = np.ones(num_items,dtype=bool)
                not_recommended[self.result[uid]] = 0
                items_not_recommended = np.nonzero(not_recommended)[0]
                if self.epsilon < np.random.rand():
                    best_item = items_not_recommended[np.argmax(items_mean_values[items_not_recommended])]
                else:
                    best_item = random.choice(items_not_recommended)
                self.result[uid].append(best_item)

            user_num_interactions = users_num_interactions[uid]
            for best_item in self.result[uid][user_num_interactions*self.interaction_size:(user_num_interactions+1)*self.interaction_size]:
                items_mean_values[best_item] = (items_mean_values[best_item]*items_count[best_item]+self.get_reward(uid,best_item))/(items_count[best_item] + 1)
                items_count[best_item] += 1

            users_num_interactions[uid] += 1
            if users_num_interactions[uid] == self.interactions:
                available_users = available_users - {uid}

        self.save_result()
