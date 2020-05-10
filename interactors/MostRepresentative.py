import numpy as np
from tqdm import tqdm
from .Interactor import Interactor
import matplotlib.pyplot as plt
import os
class MostRepresentative(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def interact(self, uids, items_latent_factors):
        super().interact()
        items_representativeness = np.sum(items_latent_factors**2,axis=1)
        
        top_iids = list(reversed(np.argsort(items_representativeness)))[:self.get_iterations()]
        num_users = len(uids)
        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            self.result[uid].extend(top_iids)
        self.save_result()
