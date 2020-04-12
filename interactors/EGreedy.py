import numpy as np
from tqdm import tqdm
from . import Interactor
import os
import random
import scipy.stats
class EGreedy(Interactor):
    def __init__(self, epsilon=0.1,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def interact(self, uids):
        super().interact()
        num_users = len(uids)
        num_items = self.consumption_matrix.shape[1]
        
        items_values = np.zeros(num_items)

        for i in tqdm(range(self.get_iterations())):
            for idx_uid in range(num_users):
                uid = uids[idx_uid]
                not_recommended = np.ones(num_items,dtype=bool)
                not_recommended[self.result[uid]] = 0
                items_not_recommended = np.nonzero(not_recommended)[0]
                if self.epsilon < np.random.rand():
                    best_item = items_not_recommended[np.argmax(items_values[items_not_recommended])]
                else:
                    best_item = random.choice(items_not_recommended)
                items_values[best_item] = max(items_values[best_item],self.get_reward(uid,best_item))
                # items_values[best_item] += self.get_reward(uid,best_item)
                self.result[uid].append(best_item)
        self.save_result()
