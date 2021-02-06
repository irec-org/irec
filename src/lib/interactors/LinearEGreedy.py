from .ICF import *
from .LinEGreedy import *
import numpy as np
import random
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import ctypes
import scipy
import joblib
from utils.PersistentDataManager import PersistentDataManager
from .LinearICF import LinearICF

class LinearEGreedy(LinearICF):

    def __init__(self, epsilon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.parameters.extend(['epsilon'])

    def train(self, train_dataset):
        super().train(train_dataset)
        self.bs = defaultdict(lambda: np.ones(self.num_lat))

    def predict(self, uid, candidate_items, num_req_items):
        b = self.bs[uid]
        A = self.As[uid]

        mean = np.dot(np.linalg.inv(A), b)

        rand = np.random.rand(min(num_req_items, len(candidate_items)))
        rand = self.epsilon > rand

        cnz = np.count_nonzero(rand)
        if cnz == min(num_req_items, len(candidate_items)):
            items_score = np.random.rand(len(candidate_items))
        else:
            items_score = mean @ self.items_means[candidate_items].T
            randind = random.sample(list(range(len(candidate_items))),
                                    k=np.count_nonzero(rand))
            items_score[randind] = np.inf

        return items_score, None

    def update(self, uid, item, reward, additional_data):
        return super().update(uid, item, reward, additional_data)
