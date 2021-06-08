from .ICF import *
from .LinEGreedy import *
import numpy as np
import random
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import ctypes
import scipy
import joblib
from lib.utils.PersistentDataManager import PersistentDataManager
from .LinearICF import LinearICF

class LinearEGreedy(LinearICF):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.epsilon = epsilon
        # self.parameters.extend(['epsilon'])

    def reset(self,observation):
        train_dataset=observation
        super().reset(train_dataset)

    def action_estimates(self,candidate_actions):
        uid=candidate_actions[0];candidate_items=candidate_actions[1]
        b = self.bs[uid]
        A = self.As[uid]

        mean = np.dot(np.linalg.inv(A), b)

        items_score = mean @ self.items_means[candidate_items].T
        # rand = np.random.rand(min(num_req_items, len(candidate_items)))
        # rand = self.epsilon > rand

        # cnz = np.count_nonzero(rand)
        # if cnz == min(num_req_items, len(candidate_items)):
            # items_score = np.random.rand(len(candidate_items))
        # else:
            # items_score = mean @ self.items_means[candidate_items].T
            # randind = random.sample(list(range(len(candidate_items))),
                                    # k=np.count_nonzero(rand))
            # items_score[randind] = np.inf

        return items_score, None

    def update(self,observation,action,reward,info):
        uid=action[0];item=action[1];additional_data=info
        return super().update(observation,action,reward,info)
