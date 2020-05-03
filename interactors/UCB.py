import numpy as np
from tqdm import tqdm
from . import Interactor
import os
import random
import scipy.stats
from collections import defaultdict
class UCB(Interactor):
    def __init__(self,c=1.0,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def interact(self, uids):
        super().interact()
        np.seterr(divide='warn')
        num_users = len(uids)
        num_items = self.consumption_matrix.shape[1]
        
        items_mean_values = np.zeros(num_items)
        items_count = np.zeros(num_items,dtype=int)

        ctime = 2

        users_num_interactions = defaultdict(int)
        available_users = set(uids)

        for i in tqdm(range(num_users*self.interactions)):
            uid = random.sample(available_users,k=1)[0]

            not_recommended = np.ones(num_items,dtype=bool)
            not_recommended[self.result[uid]] = 0
            items_not_recommended = np.nonzero(not_recommended)[0]

            items_uncertainty = self.c*np.sqrt(2*np.log(ctime,dtype=np.float64)/items_count[items_not_recommended])
            items_score = items_mean_values[items_not_recommended]+items_uncertainty
            top_items = list(reversed(np.argsort(items_score)))[:self.interaction_size]
            best_items = items_not_recommended[top_items]
            self.result[uid].extend(best_items)

            user_num_interactions = users_num_interactions[uid]
            for best_item in self.result[uid][user_num_interactions*self.interaction_size:(user_num_interactions+1)*self.interaction_size]:
                items_mean_values[best_item] = (items_mean_values[best_item]+self.get_reward(uid,best_item))/(items_count[best_item] + 1)
                items_count[best_item] += 1


            users_num_interactions[uid] += 1
            if users_num_interactions[uid] == self.interactions:
                available_users = available_users - {uid}

            ctime += 1

        self.save_result()