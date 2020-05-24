import numpy as np
from tqdm import tqdm
from .Interactor import Interactor
from .MostPopular import *
import matplotlib.pyplot as plt
import os
import scipy.sparse
from collections import defaultdict
import random
class ALMostPopular(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def interact(self, uids):
        super().interact()
        num_items = self.consumption_matrix.shape[1]
        items_popularity = MostPopular.get_items_popularity(self.consumption_matrix, uids, normalize = False)

        top_iids = list(reversed(np.argsort(items_popularity)))[:self.get_iterations()]
        num_users = len(uids)
        users_num_interactions = defaultdict(int)
        available_users = set(uids)
        for i in tqdm(range(num_users*self.interactions)):
            uid = random.sample(available_users,k=1)[0]
            not_recommended = np.ones(num_items,dtype=bool)
            not_recommended[self.results[uid]] = 0
            items_not_recommended = np.nonzero(not_recommended)[0]
            items_score = items_popularity[items_not_recommended]
            top_items = list(reversed(np.argsort(items_score)))[:self.interaction_size]
            best_items = items_not_recommended[top_items]

            items_popularity[best_items] += 1
            self.results[uid].extend(best_items)
            users_num_interactions[uid] += 1
            if users_num_interactions[uid] == self.interactions:
                available_users = available_users - {uid}

        self.save_results()
