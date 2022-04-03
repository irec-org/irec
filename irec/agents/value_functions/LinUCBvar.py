from typing import Any
import numpy as np

# import util
import scipy
import mf
from collections import defaultdict
from .MFValueFunction import MFValueFunction


class LinUCBvar(MFValueFunction):
    def __init__(self, alpha, lambda_u, zeta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha != None:
            self.alpha = alpha
        elif zeta != None:
            self.alpha = 1 + np.sqrt(np.log(2 / zeta) / 2)
        self.lambda_u = lambda_u

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (
                self.train_dataset.data[:, 2],
                (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1]),
            ),
            (self.train_dataset.num_total_users, self.train_dataset.num_total_items),
        )
        self.num_total_items = self.train_dataset.num_total_items

        mf_model = mf.SVD(num_lat=self.num_lat)
        mf_model.fit(self.train_consumption_matrix)
        self.items_weights = mf_model.items_weights
        self.num_latent_factors = len(self.items_weights[0])

        self.I = np.eye(len(self.items_weights[0]))
        self.bs: Any = defaultdict(lambda: np.ones(self.num_latent_factors))
        self.As: Any = defaultdict(lambda: self.lambda_u * self.I.copy())

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        b = self.bs[uid]
        A = self.As[uid]
        mean = np.dot(np.linalg.inv(A), b)
        items_uncertainty = self.alpha * np.sqrt(
            np.sum(
                self.items_weights[candidate_items].dot(np.linalg.inv(A))
                * self.items_weights[candidate_items],
                axis=1,
            )
        )
        items_user_similarity = mean @ self.items_weights[candidate_items].T
        items_score = items_user_similarity + items_uncertainty

        return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        max_item_latent_factors = self.items_weights[item]
        b = self.bs[uid]
        A = self.As[uid]
        A += max_item_latent_factors[:, None].dot(max_item_latent_factors[None, :])
        b += reward * max_item_latent_factors
