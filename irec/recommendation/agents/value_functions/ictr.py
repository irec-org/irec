import copy
import numpy as np
from tqdm import tqdm
import scipy.sparse
import scipy.stats
from .base import ValueFunction
from tqdm import tqdm
from numba import njit, jit
from .most_popular import *
from typing import Any


@njit
def _softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))


class _Particle:
    def __init__(self, num_users: int, num_items: int, num_lat: int):
        self.num_users = num_users
        self.num_items = num_items
        self.num_lat = num_lat
        self.alpha: float = 1
        self.beta: float = 1
        self.lambda_ = np.ones(shape=(num_lat))
        self.eta = np.ones(shape=(num_lat, num_items))
        self.mu = np.ones(shape=(num_items, num_lat))
        self.Sigma = np.array([np.identity(num_lat) for _ in range(num_items)])
        self.sigma_n_2 = scipy.stats.invgamma(self.alpha, self.beta).rvs()
        self.p = np.array(
            [np.random.dirichlet(self.lambda_[:]) for uid in range(num_users)]
        )
        self.q = np.array(
            [
                np.random.multivariate_normal(
                    self.mu[i, :], self.sigma_n_2 * self.Sigma[i, :]
                )
                for i in range(num_items)
            ]
        )
        self.Phi = np.array(
            [np.random.dirichlet(self.eta[i, :]) for i in range(num_lat)]
        )

    def p_expectations(self, uid, topic=None, reward=None):
        computed_sum = np.sum(self.lambda_[:])
        user_lambda = np.copy(self.lambda_[:])
        if reward != None:
            user_lambda[topic] += reward
            computed_sum += reward
        return user_lambda / computed_sum

    def Phi_expectations(self, item, reward=None):
        computed_sum = np.sum(self.eta, axis=1)
        item_eta = np.copy(self.eta[:, item])
        if reward != None:
            item_eta += reward
            computed_sum += reward
        return item_eta / computed_sum

    def particle_weight(self, uid, item, reward):
        norm_val = scipy.stats.norm(self.p[uid] @ self.q[item], self.sigma_n_2).pdf(
            reward
        )
        return np.sum(norm_val * self.p_expectations(uid) * self.Phi_expectations(item))

    def compute_theta(self, uid, item, reward, topic):
        return self.p_expectations(
            uid, reward=reward, topic=topic
        ) * self.Phi_expectations(item, reward=reward)

    def select_z_topic(self, uid, item, reward):
        topic = np.argmax(np.random.multinomial(1, [1 / self.num_lat] * self.num_lat))
        theta = self.compute_theta(uid, item, reward, topic)
        theta = theta / np.sum(theta)
        topic = np.argmax(np.random.multinomial(1, theta))
        return topic

    def update_parameters(self, uid, item, reward, topic):
        new_Sigma = np.linalg.inv(
            np.linalg.inv(self.Sigma[item])
            + self.p[uid][:, None] @ self.p[uid][None, :]
        )
        new_mu = new_Sigma @ (
            np.linalg.inv(self.Sigma[item]) @ self.mu[item] + self.p[uid] * reward
        )
        new_alpha = self.alpha + 1 / 2
        new_beta = self.beta + 1 / 2 * (
            self.mu[item].T @ np.linalg.inv(self.Sigma[item]) @ self.mu[item]
            + reward * reward
            - new_mu.T @ np.linalg.inv(new_Sigma) @ new_mu
        )
        new_lambda_k = self.lambda_[topic] + reward
        new_eta_k = self.eta[topic, item] + reward

        self.Sigma[item] = new_Sigma
        self.mu[item] = new_mu
        self.alpha = new_alpha
        self.beta = new_beta
        self.lambda_[topic] = new_lambda_k
        self.eta[topic, item] = new_eta_k

    def sample_random_variables(self, uid, item, topic):
        self.sigma_n_2 = scipy.stats.invgamma(self.alpha, self.beta).rvs()
        self.q[item] = np.random.multivariate_normal(
            self.mu[item], self.sigma_n_2 * self.Sigma[item]
        )
        self.p[uid] = np.random.dirichlet(self.lambda_[:])
        self.Phi[topic] = np.random.dirichlet(self.eta[topic])


class ICTRTS(ValueFunction):
    """Interactive Collaborative Topic Regression with Thompson Sampling.

    It is an interactive collaborative topic regression model that utilizes the TS
    bandit algorithm and controls the items dependency by a particle learning strategy [1]_.

    References
    ----------
    .. [1] Wang, Qing, et al. "Online interactive collaborative filtering using multi-armed
       bandit with dependent arms." IEEE Transactions on Knowledge and Data Engineering 31.8 (2018): 1569-1580.
    """

    def __init__(self, num_lat, num_particles, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
            num_particles (int):
        """
        super().__init__(*args, **kwargs)
        self.num_particles = num_particles
        self.num_lat = num_lat

    def reset(self, observation):
        """reset.

        Args:
            observation:
        """
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
        self.num_total_users = self.train_dataset.num_total_users
        particle = _Particle(self.num_total_users, self.num_total_items, self.num_lat)
        for i in tqdm(range(len(self.train_dataset.data))):
            uid = int(self.train_dataset.data[i, 0])
            item = int(self.train_dataset.data[i, 1])
            reward = self.train_dataset.data[i, 2]
            topic = particle.select_z_topic(uid, item, reward)
            particle.update_parameters(uid, item, reward, topic)
            particle.sample_random_variables(uid, item, topic)

        self.particles = [copy.deepcopy(particle) for _ in range(self.num_particles)]

    def actions_estimate(self, candidate_actions):
        """actions_estimate.

        Args:
            candidate_actions: (user id, candidate_items)

        Returns:
            numpy.ndarray:
        """
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = np.zeros(len(candidate_items))
        for particle in self.particles:
            items_score += particle.q[candidate_items, :] @ particle.p[uid, :]

        items_score /= self.num_particles
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
        weights: Any = [
            particle.particle_weight(uid, item, reward) for particle in self.particles
        ]
        weights = np.array(weights)
        weights = _softmax(weights)
        ds = np.random.choice(
            range(self.num_particles), p=weights, size=self.num_particles
        )
        new_particles = []
        for i in range(self.num_particles):
            new_particles.append(copy.deepcopy(self.particles[ds[i]]))
        self.particles = new_particles
        for particle in self.particles:
            topic = particle.select_z_topic(uid, item, reward)
            particle.update_parameters(uid, item, reward, topic)
            particle.sample_random_variables(uid, item, topic)
