import numpy as np
from .base import ValueFunction

class Random(ValueFunction):

    """Random

        This method recommends totally random items.
        
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)

    def actions_estimate(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        return np.random.rand(len(candidate_items)), None