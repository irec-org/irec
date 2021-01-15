import numpy as np
from tqdm import tqdm
from .ExperimentalInteractor import ExperimentalInteractor
import random
class Random(ExperimentalInteractor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self,train_dataset):
        super().train(train_dataset)

    def predict(self,uid,candidate_items,num_req_items):
        return np.random.rand(len(candidate_items)), None
        # uids = self.test_users
        # num_total_users = len(uids)
        # for idx_uid in tqdm(range(num_total_users)):
        #     uid = uids[idx_uid]
        #     iids= list(range(self.train_consumption_matrix.shape[1]))
        #     random.shuffle(iids)
        #     self.results[uid].extend(iids[:self.interactions*self.interaction_size])
        # self.save_results()
        
