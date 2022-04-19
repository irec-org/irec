import numpy as np
from .base import ValueFunction
import scipy.stats


class UCB(ValueFunction):
    """Upper Confidence Bound.

    It is the original UCB that calculates a confidence interval for each item
    at each iteration and tries to shrink the confidence bounds [1]_.

    References
    ----------
    .. [1] Auer, P., Cesa-Bianchi, N. & Fischer, P. Finite-time Analysis of the
       Multiarmed Bandit Problem. Machine Learning 47, 235–256 (2002).
    """

    def __init__(self, c, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
            c:
        """

        super().__init__(*args, **kwargs)
        self.c = c

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

        self.items_mean_values = np.zeros(self.num_total_items, dtype=np.float128)
        self.items_count = np.zeros(self.num_total_items, dtype=int)

        self.t = 1
        self.recent_predict = True
        for i in range(self.train_dataset.data.shape[0]):
            uid = int(self.train_dataset.data[i, 0])
            item = int(self.train_dataset.data[i, 1])
            reward = self.train_dataset.data[i, 2]
            self.update(None, (uid, item), reward, None)

    def actions_estimate(self, candidate_actions):
        """actions_estimate.

        Args:
            candidate_actions: (user id, candidate_items)

        Returns:
            numpy.ndarray:
        """

        candidate_items = candidate_actions[1]
        with np.errstate(divide="ignore"):
            items_uncertainty = self.c * np.sqrt(
                2 * np.log(self.t) / self.items_count[candidate_items]
            )
            items_score = self.items_mean_values[candidate_items] + items_uncertainty
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
        item = action[1]
        item = int(item)
        self.items_mean_values[item] = (
            self.items_mean_values[item] * self.items_count[item] + reward
        ) / (self.items_count[item] + 1)
        self.items_count[item] += 1
        if self.recent_predict:
            self.t += 1
            self.recent_predict = False