import numpy as np
from tqdm import tqdm
from . import Interactor
import os
import random
import scipy.stats
class UCB(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def interact(self, uids):
        super().interact()
        num_users = len(uids)
        num_items = self.consumption_matrix.shape[1]
        
        items_mean_values = np.zeros(num_items)
        items_count = np.zeros(num_items,dtype=int)

        ctime = 1
        for i in tqdm(range(self.get_iterations())):
            for idx_uid in range(num_users):
                uid = uids[idx_uid]
                not_recommended = np.ones(num_items,dtype=bool)
                not_recommended[self.result[uid]] = 0
                items_not_recommended = np.nonzero(not_recommended)[0]

                items_uncertainty = np.sqrt(2*np.log(ctime)/items_count[items_not_recommended])
                best_item = items_not_recommended[np.argmax(items_mean_values[items_not_recommended]+items_uncertainty)]

                items_mean_values[best_item] = (items_mean_values[best_item]+self.get_reward(uid,best_item))/(items_count[best_item] + 1)
                items_count[best_item] += 1
                self.result[uid].append(best_item)
                ctime += 1
        self.save_result()
