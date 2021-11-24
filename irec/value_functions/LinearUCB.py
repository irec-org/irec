from .ICF import ICF
import numpy as np
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import ctypes
from collections import defaultdict
import scipy
import mf
from .LinearICF import LinearICF


class LinearUCB(LinearICF):
    """LinearUCB.
    
    An adaptation of the original LinUCB (Lihong Li et al. 2010) to measure
    the latent dimensions by a PMF formulation [1]_.

    References
    ----------
    .. [1] Zhao, Xiaoxue, Weinan Zhang, and Jun Wang. "Interactive collaborative filtering." 
       Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2013.
    """
    def __init__(self, alpha, zeta=None, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
            alpha:
        """
        super().__init__(*args, **kwargs)
        if alpha != None:
            self.alpha = alpha
        elif zeta != None:
            self.alpha = 1 + np.sqrt(np.log(2 / zeta) / 2)

    def reset(self, observation):
        """reset.

        Args:
            observation: 
        """
        train_dataset = observation
        super().reset(train_dataset)

    def action_estimates(self, candidate_actions):
        """action_estimates.

        Args:
            candidate_actions: (user id, candidate_items)

        Returns:
            numpy.ndarray:
        """
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        b = self.bs[uid]
        A = self.As[uid]
        mean = np.dot(np.linalg.inv(A), b)
        cov = np.linalg.inv(A) * self.var

        items_score  = mean @ self.items_means[candidate_items].T+\
            self.alpha*np.sqrt(np.sum(self.items_means[candidate_items].dot(cov) * self.items_means[candidate_items],axis=1))

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
        return super().update(observation, action, reward, info)
