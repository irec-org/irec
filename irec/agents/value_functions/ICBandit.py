from .ExperimentalValueFunction import ExperimentalValueFunction
import numpy as np
from tqdm import tqdm
#import util
from threadpoolctl import threadpool_limits
import scipy
import scipy.stats
import ctypes
import mf
from collections import defaultdict
import value_functions
from .MFValueFunction import MFValueFunction
from .LinUCB import LinUCB


def _prediction_rule(A, b, items_weights, alpha):
    mean = np.dot(np.linalg.inv(A), b)
    items_uncertainty = alpha * np.sqrt(
        np.sum(items_weights.dot(np.linalg.inv(A)) * items_weights, axis=1))
    items_user_similarity = mean @ items_weights.T
    items_score = items_user_similarity + items_uncertainty
    return items_score


class ICLinUCB(LinUCB):
    def __init__(self, u,num_clusters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u= u
        self.num_clusters = num_clusters


    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.train_consumption_matrix

        kmeans = KMeans(self.num_clusters).fit(self.train_consumption_matrix.T)
        self.groups = list(range(self.num_clusters))

        self.groups_items = defaultdict(list)
        for uid, group in enumerate(kmeans.labels_):
            self.groups_items[group].append(uid)


    def action_estimates(self, candidate_actions):
        self.action_estimates(candidate_actions)
        return items_score, None

    def update(self, observation, action, reward, info):
        self.update(observation, action, reward, info)
        uid = action[0]
        item = action[1]
        additional_data = info
        max_item_latent_factors = self.items_weights[item]
        b = self.bs[uid]
        A = self.As[uid]
        A += max_item_latent_factors[:, None].dot(
            max_item_latent_factors[None, :])
        b += reward * max_item_latent_factors
