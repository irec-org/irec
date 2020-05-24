import numpy as np
from tqdm import tqdm
from . import Interactor, Entropy, MostPopular,LogPopEnt
import matplotlib.pyplot as plt
import scipy.stats
import os

class HELF(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_helf(items_popularity,items_entropy,num_users):

        a = np.ma.log(items_popularity).filled(0)/np.log(num_users)
        b = items_entropy/np.max(items_entropy)
        print(np.sort(b))
        print(b.min())
        print(a.min())
        print(np.sort(a))
        np.seterr('warn')
        items_helf = 2*a*b/(a+b) 
        items_helf[np.isnan(items_helf)] = 0
        return items_helf

    def interact(self):
        super().interact()
        uids = self.test_users
        # mask = np.ones(self.train_consumption_matrix.shape[0], dtype=bool)
        # mask[uids] = 0
        num_train_users = len(self.train_users)
        items_entropy = Entropy.get_items_entropy(self.train_consumption_matrix)
        items_popularity = MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)
        items_logpopent = HELF.get_items_helf(items_popularity,items_entropy,num_train_users)
        top_iids = list(reversed(np.argsort(items_logpopent)))[:self.get_iterations()]
        num_users = len(uids)
        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            self.results[uid].extend(top_iids)
        self.save_results()
