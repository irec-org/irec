from threadpoolctl import threadpool_limits
import numpy as np
from tqdm import tqdm
import scipy.sparse
from collections import defaultdict
from base import ValueFunction
from tqdm import tqdm
from numba import njit
import irec.recommendation.matrix_factorization as mf


@njit
def _softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))


class PTS(ValueFunction):
    """Particle Thompson sampling.

    It is a PMF formulation for the original TS based on a Bayesian inference around the items.
    This method also applies particle filtering to guide the exploration of items over time [1]_.

    References
    ----------
    .. [1] Wang, Qing, et al. "Online interactive collaborative filtering using multi-armed
       bandit with dependent arms." IEEE Transactions on Knowledge and Data Engineering 31.8 (2018): 1569-1580.
    """

    def __init__(self, num_lat, num_particles, var, var_u, var_v, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
            num_particles (int):
            var:
            var_u:
            var_v:
        """
        super().__init__(*args, **kwargs)
        self.num_particles = num_particles
        self.var = var
        self.var_u = var_u
        self.var_v = var_v
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

        self.particles_us = np.random.normal(
            size=(self.num_particles, self.num_total_users, self.num_lat)
        )
        self.particles_vs = np.random.normal(
            size=(self.num_particles, self.num_total_items, self.num_lat)
        )
        self.particles_var_us = np.ones(shape=(self.num_particles)) * self.var_u
        self.particles_var_vs = np.ones(shape=(self.num_particles)) * self.var_v

        self.particles_ids = np.arange(self.num_particles)
        self.items_consumed_users = defaultdict(list)
        self.items_consumed_users_rewards = defaultdict(list)
        self.users_consumed_items = defaultdict(list)
        self.users_consumed_items_rewards = defaultdict(list)
        for i in tqdm(range(len(self.train_dataset.data))):
            uid = int(self.train_dataset.data[i, 0])
            item = int(self.train_dataset.data[i, 1])
            reward = self.train_dataset.data[i, 2]
            self.users_consumed_items[uid].append(item)
            self.users_consumed_items_rewards[uid].append(reward)
            self.items_consumed_users[item].append(uid)
            self.items_consumed_users_rewards[item].append(reward)

        mf_model = mf.PMF(
            num_lat=self.num_lat, var=self.var, user_var=self.var_u, item_var=self.var_v
        )
        mf_model.fit(self.train_consumption_matrix)

        for i in range(self.num_particles):
            self.particles_us[i] = mf_model.users_weights
            self.particles_vs[i] = mf_model.items_weights


    def actions_estimate(self, candidate_actions):
        """actions_estimate.

        Args:
            candidate_actions: (user id, candidate_items)

        Returns:
            numpy.ndarray:
        """
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        particle_idx = np.random.choice(self.particles_ids)
        items_score = (
            self.particles_us[particle_idx][uid]
            @ self.particles_vs[particle_idx][candidate_items].T
        )
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
        with threadpool_limits(limits=1, user_api="blas"):
            updated_history = False

            lambdas_u_i = np.empty(
                shape=(self.num_particles, self.num_lat, self.num_lat)
            )
            zetas_u_i = np.empty(shape=(self.num_particles, self.num_lat))
            mus_u_i = np.empty(shape=(self.num_particles, self.num_lat))
            for i in range(self.num_particles):
                v_j = self.particles_vs[i][self.users_consumed_items[uid]]
                lambda_u_i = 1 / self.var * (v_j.T @ v_j) + 1 / self.particles_var_us[
                    i
                ] * np.eye(self.num_lat)
                zeta_u_i = np.sum(
                    np.multiply(
                        v_j,
                        np.array(self.users_consumed_items_rewards[uid]).reshape(-1, 1),
                    ),
                    axis=0,
                )
                lambdas_u_i[i] = lambda_u_i
                zetas_u_i[i] = zeta_u_i
                mus_u_i[i] = 1 / self.var * (np.linalg.inv(lambda_u_i) @ zeta_u_i)

            weights = np.empty(self.num_particles)
            for i in range(self.num_particles):
                lambda_u_i, mu_u_i = lambdas_u_i[i], mus_u_i[i]
                v_j = self.particles_vs[i][item, :]
                cov = 1 / self.var + np.dot(np.dot(v_j.T, lambda_u_i), v_j)
                w = scipy.stats.norm(np.dot(v_j.T, mu_u_i), cov).pdf(reward)
                weights[i] = w

            normalized_weights = _softmax(weights)
            ds = np.random.choice(
                range(self.num_particles), p=normalized_weights, size=self.num_particles
            )
            new_particles_us = np.empty(
                shape=(self.num_particles, self.num_total_users, self.num_lat)
            )
            new_particles_vs = np.empty(
                shape=(self.num_particles, self.num_total_items, self.num_lat)
            )
            new_particles_var_us = np.empty(shape=(self.num_particles))
            new_particles_var_vs = np.empty(shape=(self.num_particles))

            for i in range(self.num_particles):
                d = ds[i]
                new_particles_us[i] = self.particles_us[d]
                new_particles_vs[i] = self.particles_vs[d]
                new_particles_var_us[i] = self.particles_var_us[d]
                new_particles_var_vs[i] = self.particles_var_vs[d]

            if not updated_history:
                self.users_consumed_items[uid].append(item)
                self.users_consumed_items_rewards[uid].append(reward)
                self.items_consumed_users[item].append(uid)
                self.items_consumed_users_rewards[item].append(reward)
                updated_history = True

            for i in range(self.num_particles):
                lambda_u_i, zeta_u_i = lambdas_u_i[i], zetas_u_i[i]
                v_j = new_particles_vs[i][item, :]
                lambda_u_i += 1 / self.var * (v_j @ v_j.T)
                zeta_u_i += reward * v_j

                inv_lambda_u_i = np.linalg.inv(lambda_u_i)
                sampled_user_vector = np.random.multivariate_normal(
                    1 / self.var * (inv_lambda_u_i @ zeta_u_i), inv_lambda_u_i
                )
                new_particles_us[i][uid] = sampled_user_vector

                u_i = new_particles_us[i][self.items_consumed_users[item], :]
                lambda_v_i = 1 / self.var * (u_i.T @ u_i) + 1 / new_particles_var_vs[
                    i
                ] * np.eye(self.num_lat)

                zeta = np.sum(
                    np.multiply(
                        u_i,
                        np.array(self.items_consumed_users_rewards[item]).reshape(
                            -1, 1
                        ),
                    ),
                    axis=0,
                )
                inv_lambda_v_i = np.linalg.inv(lambda_v_i)
                item_sample_vector = np.random.multivariate_normal(
                    1 / self.var * (inv_lambda_v_i @ zeta), inv_lambda_v_i
                )
                new_particles_vs[i][item] = item_sample_vector

            if not updated_history:
                self.users_consumed_items[uid].append(item)
                self.users_consumed_items_rewards[uid].append(reward)
                self.items_consumed_users[item].append(uid)
                self.items_consumed_users_rewards[item].append(reward)
                updated_history = True

            self.particles_us = new_particles_us
            self.particles_vs = new_particles_vs
            self.particles_var_us = new_particles_var_us
            self.particles_var_vs = new_particles_var_vs
