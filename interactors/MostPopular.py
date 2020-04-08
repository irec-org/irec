import numpy as np
from tqdm import tqdm
from .Interactor import Interactor
class MostPopular(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def interact(self, uids):
        super().interact()
        num_users = len(uids)
        mask = np.ones(self.consumption_matrix.shape[0], dtype=bool)
        mask[uids] = 0
        lowest_value = np.min(self.consumption_matrix)
        top_iids = list(reversed(np.argsort(np.count_nonzero(self.consumption_matrix[mask,:]>lowest_value,axis=0))))[:self.interactions*self.interaction_size]
        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            self.result[uid].extend(top_iids)
        self.save_result()
