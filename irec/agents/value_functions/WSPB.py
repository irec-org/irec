from typing import Any
import numpy as np
import scipy.stats
import scipy.spatial
import scipy.optimize
import scipy
import mf
from collections import defaultdict
from .MFValueFunction import MFValueFunction
from value_functions.Entropy import Entropy
from value_functions.MostPopular import MostPopular
from value_functions.LogPopEnt import LogPopEnt

def _prediction_rule(A, b, items_weights, alpha):
    user_latent_factors = np.dot(np.linalg.inv(A), b)
    items_uncertainty = np.sqrt(
        np.sum(items_weights.dot(np.linalg.inv(A)) * items_weights, axis=1))
    items_user_similarity = user_latent_factors @ items_weights.T
    weighted_items_uncertainty = alpha * items_uncertainty
    user_model_items_score = items_user_similarity + weighted_items_uncertainty
    return user_model_items_score, items_user_similarity,weighted_items_uncertainty


class WSPB(MFValueFunction):
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

        items_entropy = Entropy.get_items_entropy(
            self.train_consumption_matrix)
        items_popularity = MostPopular.get_items_popularity(
            self.train_consumption_matrix, normalize=False)
        # self.items_bias = value_functions.PPELPE.get_items_ppelpe(items_popularity,items_entropy)
        self.items_bias = LogPopEnt.get_items_logpopent(
            items_popularity, items_entropy)
        print(self.items_bias.min(), self.items_bias.max())
        assert (self.items_bias.min() >= 0
                and np.isclose(self.items_bias.max(), 1))

        # regression_model = sklearn.linear_model.LinearRegression()
        res = scipy.optimize.minimize(
            lambda x, items_weights, items_bias: np.sum(
                (items_bias - x @ items_weights.T)**2),
            np.ones(self.num_latent_factors),
            args=(self.items_weights, self.items_bias),
            method='BFGS',
        )
        self.initial_b = res.x

        # print(np.corrcoef(self.items_bias,self.initial_b @ self.items_weights.T)[0,1])

        self.I = np.eye(len(self.items_weights[0]))
        self.bs: Any = defaultdict(lambda: self.initial_b.copy())
        self.As: Any = defaultdict(lambda: self.I.copy())
        # items_score = _prediction_rule(self.I, self.initial_b,
                                       # self.items_weights, self.alpha)

        # print("WSCB items score correlation with popularity:",scipy.stats.pearsonr(items_score,items_popularity),self.train_dataset.num_total_users, self.train_dataset.num_total_items)
        # print("WSCB items score correlation with entropy:",scipy.stats.pearsonr(items_score,items_entropy),self.train_dataset.num_total_users, self.train_dataset.num_total_items)
        # self.initial_b

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        b = self.bs[uid]
        A = self.As[uid]
        user_model_items_score,l1,l2 = _prediction_rule(
            A, b, self.items_weights[candidate_items], self.alpha)
        items_score = user_model_items_score
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
