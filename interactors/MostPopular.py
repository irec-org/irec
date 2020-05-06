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

        fig, ax = plt.subplots()
        ax.hist(items_popularity)
        ax.set_xlabel("Popularity")
        ax.set_ylabel("#Items")
        fig.savefig(os.path.join(self.DIRS['img'],"popularity_"+self.get_name()+".png"))

        top_iids = list(reversed(np.argsort(items_popularity)))[:self.get_iterations()]
        num_users = len(uids)
        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            self.result[uid].extend(top_iids)
        self.save_result()
