from .ExperimentalValueFunction import ExperimentalValueFunction
import numpy as np
from tqdm import tqdm
#import util
from threadpoolctl import threadpool_limits
import ctypes
import functools
from .MFValueFunction import MFValueFunction


class TinUCB(MFValueFunction):
    def __init__(self, alpha=0.2, zeta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha != None:
            self.alpha = alpha
        elif zeta != None:
            self.alpha = 1 + np.sqrt(np.log(2 / zeta) / 2)


    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (self.train_dataset.data[:, 2],
             (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1])),
            (self.train_dataset.num_total_users,
             self.train_dataset.num_total_items))
        self.num_total_items = self.train_dataset.num_total_items
        mf_model = mf.SVD(num_lat=self.num_lat)
        mf_model.fit(self.train_consumption_matrix)
        self.items_weights = mf_model.items_weights
        self.I = np.eye(len(self.items_weights[0]))
        bs = defaultdict(lambda: np.zeros(self.num_latent_factors))
        tmp_I = self.I
        As = defaultdict(lambda: tmp_I.copy())
        self.users_A_sums_history = defaultdict(lambda: [tmp_I.copy()])

    def g(self, x):
        return x

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        b = bs[uid]
        A = As[uid]
        A_weights = [
            self.g(i + 1) / self.g(len(self.users_A_sums_history[uid]))
            for i in range(len(A_sums_history))
        ]

        A = functools.reduce(
            lambda a, b: a + b,
            map(lambda x, w: x * w, zip(A_sums_history, A_weights)))

        mean = np.dot(np.linalg.inv(A), b)
        items_score = mean @ self.items_weights[candidate_items].T+\
            self.alpha*np.sqrt(np.sum(self.items_weights[candidate_items].dot(np.linalg.inv(A)) * self.items_weights[candidate_items],axis=1))
        return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        b = bs[uid]
        A = As[uid]
        max_item_latent_factors = self.items_weights[item]
        self.users_A_sums_history[uid].append(
            max_item_latent_factors[:, None].dot(
                max_item_latent_factors[None, :]))
        A += max_item_latent_factors[:, None].dot(
            max_item_latent_factors[None, :])
        b += reward * max_item_latent_factors
