import numpy as np
from tqdm import tqdm
from .Interactor import Interactor
import matplotlib.pyplot as plt
import os
import scipy.sparse
class Entropy(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def probabilities_entropy(probabilities):
        return -1*np.sum(probabilities*np.log(probabilities))

    @staticmethod
    def values_entropy(values):
        unique, counts = np.unique(values, return_counts=True)
        values_probability = counts/np.sum(counts)
        return Entropy.probabilities_entropy(values_probability)
    
    @staticmethod
    def get_items_entropy(consumption_matrix, test_uids):
        lowest_value = np.min(consumption_matrix)
        mask = np.ones(consumption_matrix.shape[0], dtype=bool)
        mask[test_uids] = 0
        items_entropy = np.zeros(consumption_matrix.shape[1])
        is_spmatrix = isinstance(consumption_matrix,scipy.sparse.spmatrix)
        if is_spmatrix:
            consumption_matrix = scipy.sparse.csc_matrix(consumption_matrix)
        # consumption_matrix=consumption_matrix.transpose()
        # import time
        for iid in range(consumption_matrix.shape[1]):
            # stime = time.time()
            if is_spmatrix:
                # iid_ratings = consumption_matrix[mask].data
                # iid_ratings = consumption_matrix[mask,:][:,iid].data
                # consumption_matrix.getcol(iid)
                # iid_ratings = consumption_matrix[:,iid][mask].data
                iid_ratings = consumption_matrix[:,iid].A.flatten()[mask]
                iid_ratings = iid_ratings[iid_ratings > lowest_value]
            else:
                iid_ratings = consumption_matrix[mask,iid]
                iid_ratings = iid_ratings[iid_ratings > lowest_value]
            # print(f"Elapsed time: {time.time()-stime}")
            # unique, counts = np.unique(iid_ratings, return_counts=True)
            # ratings_probability = counts/np.sum(counts)
            # items_entropy[iid] = -1*np.sum(ratings_probability*np.log(ratings_probability))
            items_entropy[iid] = Entropy.values_entropy(iid_ratings)
        return items_entropy

    def interact(self, uids):
        super().interact()
        num_users = len(uids)
        items_entropy = self.get_items_entropy(self.consumption_matrix, uids)
        fig, ax = plt.subplots()
        ax.hist(items_entropy,color='k')
        ax.set_xlabel("Entropy")
        ax.set_ylabel("#Items")
        fig.savefig(os.path.join(self.DIRS['img'],"entropy_"+self.get_name()+".png"))
        
        top_iids = list(reversed(np.argsort(items_entropy)))[:self.get_iterations()]

        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            self.results[uid].extend(top_iids)
        self.save_results()
