import numpy as np
from tqdm import tqdm
from .ExperimentalValueFunction import ExperimentalValueFunction
import random


class Random(ExperimentalValueFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        return np.random.rand(len(candidate_items)), None
        # uids = self.test_users
        # num_total_users = len(uids)
        # for idx_uid in tqdm(range(num_total_users)):
        #     uid = uids[idx_uid]
        #     iids= list(range(self.train_consumption_matrix.shape[1]))
        #     random.shuffle(iids)
        #     self.results[uid].extend(iids[:self.interactions*self.interaction_size])
        # self.save_results()
