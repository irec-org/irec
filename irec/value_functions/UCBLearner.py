import numpy as np
import scipy.spatial
import mf
from .MFValueFunction import MFValueFunction
from .Entropy import Entropy
from .MostPopular import MostPopular
from .LogPopEnt import LogPopEnt
from collections import defaultdict
import scipy.sparse


class UCBLearner(MFValueFunction):
    def __init__(self,
                 alpha=1.0,
                 stop=None,
                 weight_method='change',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.weight_method = weight_method
        self.stop = stop


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
        self.num_latent_factors = len(mf_model.items_weights[0])

        items_entropy = Entropy.get_items_entropy(
            self.train_consumption_matrix)
        items_popularity = MostPopular.get_items_popularity(
            self.train_consumption_matrix, normalize=False)
        self.items_bias = LogPopEnt.get_items_logpopent(
            items_popularity, items_entropy)

        assert (self.items_bias.min() >= 0 and self.items_bias.max() == 1)

        self.I = np.eye(len(self.items_weights[0]))
        self.bs = defaultdict(lambda: np.zeros(self.num_latent_factors))
        self.As = defaultdict(lambda: self.I.copy())

        self.users_latent_factors_history = defaultdict(
            lambda: np.empty(shape=(0, self.num_latent_factors)))
        self.users_num_correct_items_history = defaultdict(lambda: [0])
        self.users_similarity_score = defaultdict(lambda: [0])
        self.users_distance_history = defaultdict(lambda: [])
        self.users_global_model_weights = defaultdict(lambda: [])

    def get_global_model_weight(self, user_latent_factors_history,
                                num_correct_items_history, distance_history):
        if self.weight_method == 'stop':
            b = self.stop
            a = min(np.sum(num_correct_items_history), b)
            return (1 - np.round(pow(2, a) / pow(2, b), 3))
        elif self.weight_method == 'change':
            if len(user_latent_factors_history) == 0:
                return 1
            times_with_reward = np.nonzero(num_correct_items_history)[0]
            if len(times_with_reward) < 2:
                return 1
            res = scipy.spatial.distance.cosine(
                user_latent_factors_history[times_with_reward][-1],
                user_latent_factors_history[times_with_reward][-2])
            distance_history.append(res)
            res = res / np.max(distance_history)
            # res = (res+1)/2
            return res
        else:
            raise RuntimeError

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        b = self.bs[uid]
        A = self.As[uid]

        self.users_latent_factors_history[uid]
        num_correct_items_history = self.users_num_correct_items_history[uid]
        similarity_score = self.users_similarity_score[uid]
        distance_history = self.users_distance_history[uid]
        global_model_weights = self.users_global_model_weights[uid]

        user_latent_factors = np.dot(np.linalg.inv(A), b)
        self.users_latent_factors_history[uid] = np.vstack(
            [self.users_latent_factors_history[uid], user_latent_factors])
        global_model_weight = self.get_global_model_weight(
            self.users_latent_factors_history[uid], num_correct_items_history,
            distance_history)
        global_model_weights.append(global_model_weight)
        items_uncertainty = np.sqrt(
            np.sum(self.items_weights[candidate_items].dot(np.linalg.inv(A)) *
                   self.items_weights[candidate_items],
                   axis=1))
        items_user_similarity = user_latent_factors @ self.items_weights[
            candidate_items].T
        user_model_items_score = items_user_similarity + self.alpha * items_uncertainty
        global_model_items_score = self.items_bias[candidate_items]
        user_model_items_score_min = np.min(user_model_items_score)
        user_model_items_score_max = np.max(user_model_items_score)
        if user_model_items_score_max - user_model_items_score_min != 0:
            global_model_items_score = global_model_items_score * (
                user_model_items_score_max -
                user_model_items_score_min) + user_model_items_score_min

        items_score = (
            1 - global_model_weight
        ) * user_model_items_score + global_model_weight * global_model_items_score
        return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        # additional_data = info
        max_item_latent_factors = self.items_weights[item]
        b = self.bs[uid]
        A = self.As[uid]
        A += max_item_latent_factors[:, None].dot(
            max_item_latent_factors[None, :])
        b += reward * max_item_latent_factors
        if reward > min(self.train_dataset.rate_domain):
            num_correct_items_history = self.users_num_correct_items_history[
                uid]
            num_correct_items_history[-1] += 1
