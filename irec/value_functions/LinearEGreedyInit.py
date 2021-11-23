from .ICF import ICF
import numpy as np
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import ctypes
from collections import defaultdict
import scipy
import mf
from .LinearEGreedy import *
import value_functions


class LinearEGreedyInit(LinearEGreedy):
    def __init__(self, init, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init = init


    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)

        if self.init == 'entropy':
            items_entropy = value_functions.Entropy.get_items_entropy(
                self.train_consumption_matrix)
            self.items_bias = items_entropy
        elif self.init == 'popularity':
            items_popularity = value_functions.MostPopular.get_items_popularity(
                self.train_consumption_matrix, normalize=False)
            self.items_bias = items_popularity
        elif self.init == 'logpopent':
            items_entropy = value_functions.Entropy.get_items_entropy(
                self.train_consumption_matrix)
            items_popularity = value_functions.MostPopular.get_items_popularity(
                self.train_consumption_matrix, normalize=False)
            self.items_bias = value_functions.LogPopEnt.get_items_logpopent(
                items_popularity, items_entropy)
        elif self.init == 'rand_popularity':
            items_popularity = value_functions.MostPopular.get_items_popularity(
                self.train_consumption_matrix, normalize=False)
            items_popularity[np.argsort(items_popularity)[::-1][100:]] = 0
            self.items_bias = items_popularity
        elif self.init == 'random':
            self.items_bias = np.random.rand(
                self.train_dataset.num_total_items)

        self.items_bias = self.items_bias - np.min(self.items_bias)
        self.items_bias = self.items_bias / np.max(self.items_bias)

        assert (self.items_bias.min() >= 0
                and np.isclose(self.items_bias.max(), 1))

        res = scipy.optimize.minimize(
            lambda x, items_means, items_bias: np.sum(
                (items_bias - x @ items_means.T)**2),
            np.ones(self.num_latent_factors),
            args=(self.items_means, self.items_bias),
            method='BFGS',
        )
        self.initial_b = res.x

        print(
            np.corrcoef(self.items_bias,
                        self.initial_b @ self.items_means.T)[0, 1])

        self.bs = defaultdict(lambda: self.initial_b.copy())


class LinearEGreedyEntropy(LinearEGreedyInit):
    def __init__(self, *args, **kwargs):
        super().__init__(init='entropy', *args, **kwargs)


class LinearEGreedyPopularity(LinearEGreedyInit):
    def __init__(self, *args, **kwargs):
        super().__init__(init='popularity', *args, **kwargs)


class LinearEGreedyRandPopularity(LinearEGreedyInit):
    def __init__(self, *args, **kwargs):
        super().__init__(init='rand_popularity', *args, **kwargs)


class LinearEGreedyRandom(LinearEGreedyInit):
    def __init__(self, *args, **kwargs):
        super().__init__(init='random', *args, **kwargs)


class LinearEGreedyLogPopEnt(LinearEGreedyInit):
    def __init__(self, *args, **kwargs):
        super().__init__(init='logpopent', *args, **kwargs)
