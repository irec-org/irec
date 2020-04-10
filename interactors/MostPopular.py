import numpy as np
from tqdm import tqdm
from .Interactor import Interactor
class MostPopular(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_most_popular(consumption_matrix,test_uids):
        uids = test_uids
        num_users = len(uids)
        mask = np.ones(consumption_matrix.shape[0], dtype=bool)
        mask[uids] = 0
        lowest_value = np.min(consumption_matrix)
        top_iids = list(reversed(np.argsort(np.count_nonzero(consumption_matrix[mask,:]>lowest_value,axis=0))))
        return top_iids

    def interact(self, uids):
        super().interact()
        top_iids = self.get_most_popular(self.consumption_matrix, uids)[:self.get_iterations()]
        num_users = len(uids)
        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            self.result[uid].extend(top_iids)
        self.save_result()
