import numpy as np
from tqdm import tqdm
#import util
from threadpoolctl import threadpool_limits
import ctypes
import scipy.spatial
import matplotlib.pyplot as plt
import os
import pickle
import sklearn
import scipy.optimize
import scipy
import mf
from collections import defaultdict
from .MFValueFunction import MFValueFunction
import value_functions


def _ucb(x, A, alpha, items_weights):
    return x @ items_weights.T + alpha * np.sqrt(
        np.sum(items_weights.dot(np.linalg.inv(A)) * items_weights, axis=1))


class OurMethod3(MFValueFunction):
    def __init__(self, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha


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
        self.num_latent_factors = len(self.items_weights[0])

        items_entropy = value_functions.Entropy.get_items_entropy(
            self.train_consumption_matrix)
        items_popularity = value_functions.MostPopular.get_items_popularity(
            self.train_consumption_matrix, normalize=False)
        self.items_bias = value_functions.LogPopEnt.get_items_logpopent(
            items_popularity, items_entropy)
        assert (self.items_bias.min() >= 0
                and np.isclose(self.items_bias.max(), 1))
        self.initial_b = np.ones(self.num_lat)
        self.I = np.eye(len(self.items_weights[0]))
        self.bs = defaultdict(lambda: self.initial_b.copy())
        self.As = defaultdict(lambda: self.I.copy())
        self.num_recommended_items = defaultdict(int)

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        if self.num_recommended_items[uid] < 20:
            items_score = self.items_bias[candidate_items]
        else:
            b = self.bs[uid]
            A = self.As[uid]
            user_latent_factors = np.dot(np.linalg.inv(A), b)
            items_uncertainty = np.sqrt(
                np.sum(
                    self.items_weights[candidate_items].dot(np.linalg.inv(A)) *
                    self.items_weights[candidate_items],
                    axis=1))
            items_user_similarity = user_latent_factors @ self.items_weights[
                candidate_items].T
            user_model_items_score = items_user_similarity + self.alpha * items_uncertainty
            items_score = user_model_items_score

        self.num_recommended_items[uid] += num_req_items
        return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        max_item_latent_factors = self.items_weights[item]
        b = self.bs[uid]
        A = self.As[uid]
        A += max_item_latent_factors[:, None].dot(
            max_item_latent_factors[None, :])
        b += reward * max_item_latent_factors
