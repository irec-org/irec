import numpy as np
from .experimental_valueFunction import ExperimentalValueFunction

class Random(ExperimentalValueFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        return np.random.rand(len(candidate_items)), None