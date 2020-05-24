import numpy as np
from tqdm import tqdm
from .Interactor import Interactor
import matplotlib.pyplot as plt
import os
class MostRepresentative(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_representativeness(items_latent_factors):
        return np.sum(items_latent_factors**2,axis=1)

    def interact(self, items_latent_factors):
        super().interact()
        uids = self.test_users
        items_representativeness = self.get_items_representativeness(items_latent_factors)
        top_iids = list(reversed(np.argsort(items_representativeness)))[:self.get_iterations()]
        num_users = len(uids)
        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            self.results[uid].extend(top_iids)
        self.save_results()
