from .ICF import ICF
from .LinearICF import LinearICF
import numpy as np
from tqdm import tqdm
#import util
from threadpoolctl import threadpool_limits
import scipy.optimize
import ctypes
from collections import defaultdict
import scipy
import mf
import value_functions


class GLM_UCB(LinearICF):
    """Generalized Linear Model Bandit-Upper Confidence Bound.
    
    It follows a similar process as Linear UCB based on the PMF formulation, but it also
    adds a sigmoid form in the exploitation step and makes a time-dependent exploration [1]_.

    References
    ----------
    .. [1] Zhao, Xiaoxue, Weinan Zhang, and Jun Wang. "Interactive collaborative filtering." 
       Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2013.
    """
    def __init__(self, c=1.0, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
            c (float):
        """
        super().__init__(*args, **kwargs)
        self.c = c


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def p(self, x):
        return x

    def reset(self, observation):
        """reset.

        Args:
            observation: 
        """ 
        train_dataset = observation
        super().reset(train_dataset)
        self.users_rec_rewards = defaultdict(list)
        self.users_rec_items_means = defaultdict(list)
        self.p_vals = dict()
        np.seterr(under="ignore")
        self.recent_predict = True
        self.t = 0

    def error_user_weight_function(self, p, u_rec_rewards, u_rec_items_means):
        return np.sum(
            np.array([(u_rec_rewards[t] - self.p(p.T @ u_rec_items_means[t])) *
                      u_rec_items_means[t]
                      for t in range(0, len(u_rec_items_means))]), 0)

    def action_estimates(self, candidate_actions):
        """action_estimates.

        Args:
            candidate_actions: (user id, candidate_items)
        
        Returns:
            numpy.ndarray:
        """
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        A = self.As[uid]
        if len(self.users_rec_items_means[uid]) == 0:
            # self.p_vals[uid] = np.zeros(self.num_latent_factors)
            # self.p_vals[uid] = self.bs[uid]+1
            # print(self.bs[uid].dtype)
            self.p_vals[uid] = self.bs[uid]
            # self.p_vals[uid] = np.array(self.bs[uid],dtype=float)
        else:
            # self.p_vals[uid] = np.array(self.p_vals[uid],dtype=float)
            self.p_vals[uid] = scipy.optimize.root(
                self.error_user_weight_function, self.p_vals[uid],
                (self.users_rec_rewards[uid],
                 self.users_rec_items_means[uid])).x
        cov = np.linalg.inv(A) * self.var
        items_score = self.p(self.p_vals[uid][None,:] @ self.items_means[candidate_items].T) +\
            self.c * np.sqrt(np.log(self.t+1)) *\
            np.sqrt(np.sum(self.items_means[candidate_items].dot(cov) *\
                           self.items_means[candidate_items],axis=1))
        items_score = items_score.flatten()
        self.recent_predict = True
        return items_score, None

    def update(self, observation, action, reward, info):
        """update.

        Args:
            observation:
            action: (user id, item)
            reward (float): reward
            info: 
        """
        uid = action[0]
        item = action[1]
        additional_data = info
        max_item_mean = self.items_means[item, :]
        self.users_rec_rewards[uid].append(reward)
        self.users_rec_items_means[uid].append(max_item_mean)
        self.As[uid] += max_item_mean[:, None].dot(max_item_mean[None, :])
        if self.recent_predict:
            self.t += 1
            self.recent_predict = False


class GLM_UCBInit(GLM_UCB):
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


class GLM_UCBEntropy(GLM_UCBInit):
    def __init__(self, *args, **kwargs):
        super().__init__(init='entropy', *args, **kwargs)


class GLM_UCBPopularity(GLM_UCBInit):
    def __init__(self, *args, **kwargs):
        super().__init__(init='popularity', *args, **kwargs)


class GLM_UCBRandPopularity(GLM_UCBInit):
    def __init__(self, *args, **kwargs):
        super().__init__(init='rand_popularity', *args, **kwargs)


class GLM_UCBRandom(GLM_UCBInit):
    def __init__(self, *args, **kwargs):
        super().__init__(init='random', *args, **kwargs)


class GLM_UCBLogPopEnt(GLM_UCBInit):
    def __init__(self, *args, **kwargs):
        super().__init__(init='logpopent', *args, **kwargs)
