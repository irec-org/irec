from .ICF import ICF
import numpy as np
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import ctypes
from collections import defaultdict
import joblib
import scipy
import mf
from lib.utils.PersistentDataManager import PersistentDataManager
from .LinearICF import LinearICF


class LinearUCB(LinearICF):
    def __init__(self, alpha, zeta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha != None:
            self.alpha = alpha
        elif zeta != None:
            self.alpha = 1 + np.sqrt(np.log(2 / zeta) / 2)

        self.parameters.extend(['alpha'])

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        b = self.bs[uid]
        A = self.As[uid]
        mean = np.dot(np.linalg.inv(A), b)
        cov = np.linalg.inv(A) * self.var

        items_score  = mean @ self.items_means[candidate_items].T+\
            self.alpha*np.sqrt(np.sum(self.items_means[candidate_items].dot(cov) * self.items_means[candidate_items],axis=1))

        return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        return super().update(observation, action, reward, info)
