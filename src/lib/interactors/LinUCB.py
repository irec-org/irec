from .ExperimentalInteractor import ExperimentalInteractor
import numpy as np
from tqdm import tqdm
#import util
from threadpoolctl import threadpool_limits
import scipy
import ctypes
import mf
from collections import defaultdict
from .MFInteractor import MFInteractor


class LinUCB(MFInteractor):

    def __init__(self, alpha, zeta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha != None:
            self.alpha = alpha
        elif zeta != None:
            self.alpha = 1 + np.sqrt(np.log(2 / zeta) / 2)
        self.parameters.extend(['alpha'])

    def train(self, train_dataset):
        super().train(train_dataset)
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
        self.num_latent_factors = len(self.items_weights[0])

        self.I = np.eye(len(self.items_weights[0]))
        self.bs = defaultdict(lambda: np.ones(self.num_latent_factors))
        self.As = defaultdict(lambda: self.I.copy())

    def predict(self, uid, candidate_items, num_req_items):
        b = self.bs[uid]
        A = self.As[uid]
        mean = np.dot(np.linalg.inv(A), b)
        items_uncertainty = self.alpha * np.sqrt(
            np.sum(self.items_weights[candidate_items].dot(np.linalg.inv(A)) *
                   self.items_weights[candidate_items],
                   axis=1))
        items_user_similarity = mean @ self.items_weights[candidate_items].T
        items_score = items_user_similarity + items_uncertainty
        # best_item = candidate_items[np.argmax(items_score)]
        # print(uid,best_item,items_user_similarity[best_item],items_uncertainty[best_item])
        return items_score, None

    def update(self, uid, item, reward, additional_data):
        max_item_latent_factors = self.items_weights[item]
        b = self.bs[uid]
        A = self.As[uid]
        A += max_item_latent_factors[:,
                                     None].dot(max_item_latent_factors[None, :])
        b += reward * max_item_latent_factors
