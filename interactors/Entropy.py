import numpy as np
from tqdm import tqdm
from .Interactor import Interactor
import matplotlib.pyplot as plt
import os
class Entropy(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)


    @staticmethod
    def get_items_entropy(consumption_matrix, test_uids):
        lowest_value = np.min(consumption_matrix)
        mask = np.ones(consumption_matrix.shape[0], dtype=bool)
        mask[test_uids] = 0
        items_entropy = np.zeros(consumption_matrix.shape[1])
        for iid in range(consumption_matrix.shape[1]):
            iid_ratings = consumption_matrix[mask,iid]
            iid_ratings = iid_ratings[iid_ratings > lowest_value]
            unique, counts = np.unique(iid_ratings, return_counts=True)
            ratings_probability = counts/np.sum(counts)
            items_entropy[iid] = -1*np.sum(ratings_probability*np.log(ratings_probability))
        return items_entropy

    def interact(self, uids):
        super().interact()
        num_users = len(uids)
        items_entropy = self.get_items_entropy(self.consumption_matrix, uids)

        plt.hist(items_entropy)
        plt.xlabel("Entropy")
        plt.ylabel("#Items")
        plt.savefig(os.path.join(self.DIRS['img'],"entropy_"+self.get_name()+".png"))
        plt.clf()
        
        top_iids = list(reversed(np.argsort(items_entropy)))[:self.get_iterations()]
        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            self.result[uid].extend(top_iids)
        self.save_result()
