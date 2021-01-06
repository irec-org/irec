import numpy as np
from tqdm import tqdm
from . import Interactor
import os
import random
import scipy.stats
from collections import defaultdict
class UCB(ExperimentalInteractor):
    def __init__(self,c=1.0,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def interact(self):
        super().interact()
        uids = self.test_users
        np.seterr(divide='warn')
        num_users = len(uids)
        num_items = self.train_consumption_matrix.shape[1]
        
        items_mean_values = np.zeros(num_items,dtype=np.float128)
        items_count = np.zeros(num_items,dtype=int)

        ctime = 1

        users_num_interactions = defaultdict(int)
        available_users = set(uids)

        mask = np.ones(self.train_consumption_matrix.shape[0], dtype=bool)
        mask[uids] = 0
        items_mean_values = np.mean(self.train_consumption_matrix[mask],axis=0).A.flatten()
        items_count += self.train_consumption_matrix[mask].shape[0]
        ctime += np.prod(self.train_consumption_matrix[mask].shape)

        for i in tqdm(range(num_users*self.interactions)):
            uid = random.sample(available_users,k=1)[0]

            not_recommended = np.ones(num_items,dtype=bool)
            not_recommended[self.results[uid]] = 0
            items_not_recommended = np.nonzero(not_recommended)[0]

            items_uncertainty = self.c*np.sqrt(2*np.log(ctime)/items_count[items_not_recommended])
            items_score = items_mean_values[items_not_recommended]+items_uncertainty

            # if ctime % 5000 <= 5:
            #     print(np.max(items_uncertainty))
            #     print(np.max(items_mean_values[items_not_recommended]))
            top_items = list(reversed(np.argsort(items_score)))[:self.interaction_size]
            best_items = items_not_recommended[top_items]
            self.results[uid].extend(best_items)

            user_num_interactions = users_num_interactions[uid]
            for best_item in self.results[uid][user_num_interactions*self.interaction_size:(user_num_interactions+1)*self.interaction_size]:
                items_mean_values[best_item] = (items_mean_values[best_item]*items_count[best_item]+self.get_reward(uid,best_item))/(items_count[best_item] + 1)
                items_count[best_item] += 1


            users_num_interactions[uid] += 1
            if users_num_interactions[uid] == self.interactions:
                available_users = available_users - {uid}

            ctime += self.interaction_size

        self.save_results()
