from .icf import *
import numpy as np
from .linear_icf import LinearICF


class LinearEGreedy(LinearICF):
    """Linear Epsilon Greedy.
    
    A linear exploitation of the items latent factors defined by a PMF
    formulation that also explore random items with probability ε [1]_.

    References
    ----------
    .. [1 ]Zhao, Xiaoxue, Weinan Zhang, and Jun Wang. "Interactive collaborative filtering." 
       Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2013.
    """
    def __init__(self, num_lat, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
        """
        super().__init__(num_lat=num_lat, *args, **kwargs)
        self.num_lat = num_lat

    def reset(self, observation):
        """reset.

        Args:
            observation: 
        """                
        train_dataset = observation
        super().reset(train_dataset)

    def actions_estimate(self, candidate_actions):
        """actions_estimate.

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

        items_score = mean @ self.items_means[candidate_items].T
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
