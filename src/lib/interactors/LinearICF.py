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


class LinearICF(ICF):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (self.train_dataset.data[:, 2],
             (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1])),
            (self.train_dataset.num_total_users,
             self.train_dataset.num_total_items))
        self.num_total_items = self.train_dataset.num_total_items
        mf_model = mf.ICFPMFS(self.iterations,
                              self.var,
                              self.user_var,
                              self.item_var,
                              self.stop_criteria,
                              num_lat=self.num_lat)
        mf_model_id = joblib.hash(
            (mf_model.get_id(), self.train_consumption_matrix))
        pdm = PersistentDataManager('state_save')
        if pdm.file_exists(mf_model_id):
            mf_model = pdm.load(mf_model_id)
        else:
            mf_model.fit(self.train_consumption_matrix)
            pdm.save(mf_model_id, mf_model)

        self.items_means = mf_model.items_means

        self.num_latent_factors = len(self.items_means[0])

        self.I = np.eye(self.num_latent_factors)
        self.bs = defaultdict(lambda: np.zeros(self.num_latent_factors))
        self.As = defaultdict(lambda: self.get_user_lambda() * self.I)

    def predict(self, uid, candidate_items, num_req_items):
        b = self.bs[uid]
        A = self.As[uid]
        mean = np.dot(np.linalg.inv(A), b)
        cov = np.linalg.inv(A) * self.var

        items_score  = mean @ self.items_means[candidate_items].T

        return items_score, None

    def update(self, uid, item, reward, additional_data):
        max_item_mean = self.items_means[item]
        b = self.bs[uid]
        A = self.As[uid]
        A += max_item_mean[:, None].dot(max_item_mean[None, :])
        b += reward * max_item_mean
