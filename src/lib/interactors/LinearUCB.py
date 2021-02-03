from .ICF import ICF
import numpy as np
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import ctypes
from collections import defaultdict
import joblib
import scipy
import mf
from utils.PersistentDataManager import PersistentDataManager
from .LinearICF import LinearICF


class LinearUCB(LinearICF):

    def __init__(self, alpha, zeta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha != None:
            self.alpha = alpha
        elif zeta != None:
            self.alpha = 1 + np.sqrt(np.log(2 / zeta) / 2)

        self.parameters.extend(['alpha'])

    def train(self, train_dataset):
        super().train(train_dataset)

    def predict(self, uid, candidate_items, num_req_items):
        b = self.bs[uid]
        A = self.As[uid]
        mean = np.dot(np.linalg.inv(A), b)
        cov = np.linalg.inv(A) * self.var

        items_score  = mean @ self.items_means[candidate_items].T+\
            self.alpha*np.sqrt(np.sum(self.items_means[candidate_items].dot(cov) * self.items_means[candidate_items],axis=1))

        return items_score, None

    def update(self, uid, item, reward, additional_data):
        return super().update(uid, item, reward, additional_data)
