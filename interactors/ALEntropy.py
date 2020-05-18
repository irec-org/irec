import numpy as np
from tqdm import tqdm
from .Interactor import *
from .Entropy import *
import matplotlib.pyplot as plt
import os
import scipy.sparse
from collections import defaultdict
import random
class ALEntropy(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def interact(self, uids):
        super().interact()
        num_items = self.consumption_matrix.shape[1]
        if self.is_spmatrix:
            unique_values = np.unique(self.consumption_matrix.data)
        else:
            unique_values = np.unique(self.consumption_matrix[self.consumption_matrix>self.lowest_value])
        num_unique_values = len(unique_values)
        items_ratings = np.zeros((num_items,num_unique_values))
        unique_values_ids = dict(zip(unique_values,list(range(num_unique_values))))

        num_users = len(uids)
        users_num_interactions = defaultdict(int)
        available_users = set(uids)
        for i in tqdm(range(num_users*self.interactions)):
            uid = random.sample(available_users,k=1)[0]
            not_recommended = np.ones(num_items,dtype=bool)
            not_recommended[self.result[uid]] = 0
            items_not_recommended = np.nonzero(not_recommended)[0]
            items_score =  [Entropy.values_entropy(items_ratings[iid])
                            for iid
                            in items_not_recommended]
            top_items = list(reversed(np.argsort(items_score)))[:self.interaction_size]
            best_items = items_not_recommended[top_items]

            for best_item in best_items:
                reward = self.get_reward(uid,best_item)
                if reward > self.lowest_value:
                    items_ratings[best_item,unique_values_ids[reward]] = reward

            self.result[uid].extend(best_items)
            users_num_interactions[uid] += 1
            if users_num_interactions[uid] == self.interactions:
                available_users = available_users - {uid}

        self.save_result()
