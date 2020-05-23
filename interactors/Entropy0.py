import numpy as np
from tqdm import tqdm
from .Interactor import Interactor
import matplotlib.pyplot as plt
import os
import scipy.sparse
class Entropy0(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def probabilities_entropy(probabilities):
        return -1*np.sum(probabilities*np.log(probabilities))

    @staticmethod
    def values_entropy(values):
        unique, counts = np.unique(values, return_counts=True)
        values_probability = counts/np.sum(counts)
        return Entropy0.probabilities_entropy(values_probability)
    
    @staticmethod
    def get_items_entropy(consumption_matrix, test_uids):
        lowest_value = np.min(consumption_matrix)
        mask = np.ones(consumption_matrix.shape[0], dtype=bool)
        mask[test_uids] = 0
        items_entropy = np.zeros(consumption_matrix.shape[1])
        is_spmatrix = isinstance(consumption_matrix,scipy.sparse.spmatrix)
        if is_spmatrix:
            consumption_matrix = scipy.sparse.csc_matrix(consumption_matrix)
        for iid in range(consumption_matrix.shape[1]):
            if is_spmatrix:
                iid_ratings = consumption_matrix[:,iid].A.flatten()[mask]
            else:
                raise RuntimeError
            items_entropy[iid] = Entropy0.values_entropy(iid_ratings)
        return items_entropy

    def interact(self, uids):
        super().interact()
        num_users = len(uids)
        items_entropy = self.get_items_entropy(self.consumption_matrix, uids)
        fig, ax = plt.subplots()
        ax.hist(items_entropy,color='k')
        ax.set_xlabel("Entropy0")
        ax.set_ylabel("#Items")
        fig.savefig(os.path.join(self.DIRS['img'],"entropy0_"+self.get_name()+".png"))
        
        top_iids = list(reversed(np.argsort(items_entropy)))[:self.get_iterations()]

        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            self.result[uid].extend(top_iids)
        self.save_result()
