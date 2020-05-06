import numpy as np
from tqdm import tqdm
from .Interactor import Interactor
import matplotlib.pyplot as plt
import os
class MostPopular(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_popularity(consumption_matrix, test_uids, normalize=True):
        uids = test_uids
        num_users = len(uids)
        mask = np.ones(consumption_matrix.shape[0], dtype=bool)
        mask[uids] = 0
        lowest_value = np.min(consumption_matrix)
        items_popularity = np.count_nonzero(consumption_matrix[mask,:]>lowest_value,axis=0)
        if normalize:
            items_popularity = items_popularity/consumption_matrix.shape[0]
        return items_popularity

    def interact(self, uids):
        super().interact()
        items_popularity = self.get_items_popularity(self.consumption_matrix, uids)

        plt.hist(items_popularity)
        plt.xlabel("Popularity")
        plt.ylabel("#Items")
        plt.savefig(os.path.join(self.DIRS['img'],"popularity_"+self.get_name()+".png"))
        plt.clf()

        top_iids = list(reversed(np.argsort(items_popularity)))[:self.get_iterations()]

        print(top_iids[:20])

        num_users = len(uids)
        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            self.result[uid].extend(top_iids)
        self.save_result()
