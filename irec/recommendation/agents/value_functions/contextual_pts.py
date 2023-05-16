from irec.recommendation.agents.value_functions import Entropy
from irec.recommendation.agents.value_functions import MostPopular
from irec.recommendation.agents.value_functions.log_pop_ent import LogPopEnt
from irec.recommendation.agents.value_functions import PTS
import scipy.sparse
import numpy as np


class ContextualPTS(PTS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)

        items_entropy = Entropy.get_items_entropy(
            self.train_consumption_matrix)
        items_popularity = MostPopular.get_items_popularity(
            self.train_consumption_matrix, normalize=False)
        self.items_bias = LogPopEnt.get_items_logpopent(
            items_popularity, items_entropy)

        self.items_bias = self.items_bias - np.min(self.items_bias)
        self.items_bias = self.items_bias / np.max(self.items_bias)
        assert self.items_bias.min() >= 0 and np.isclose(self.items_bias.max(), 1)
        res = scipy.optimize.minimize(
            lambda x, items_means, items_bias: np.sum(
                (items_bias - x @ items_means.T) ** 2
            ),
            np.ones(self.num_lat),
            args=(self.particles_vs[0], self.items_bias),
            method="BFGS",
        )
        self.initial_user_factors = res.x
        not_train_users = set(list(range(self.train_dataset.num_total_users))) - {
            int(self.train_dataset.data[i, 0])
            for i in range(len(self.train_dataset.data))
        }
        for i in range(self.num_particles):
            for uid in not_train_users:
                self.particles_us[i][uid] = self.initial_user_factors
