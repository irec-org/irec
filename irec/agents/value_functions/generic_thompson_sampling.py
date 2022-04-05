import numpy as np
import numbers
from .experimental_valueFunction import ExperimentalValueFunction
from collections import defaultdict


def _create_attribute_with_check(value):
    if isinstance(value, numbers.Number):
        return defaultdict(lambda: value)
    elif isinstance(value, dict):
        return defaultdict(lambda: 1, value)


class GenericThompsonSampling(ExperimentalValueFunction):
    def __init__(self, alpha_0, beta_0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        #

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
      
        self.alphas = _create_attribute_with_check(self.alpha_0)
        self.betas = _create_attribute_with_check(self.beta_0)

    def action_estimates(self, candidate_actions):
        items_score = np.random.beta(
            [self.alphas[ca] for ca in candidate_actions],
            [self.betas[ca] for ca in candidate_actions],
        )
        return items_score, None

    def update(self, observation, action, reward, info):
        reward = 1 if (reward >= 4) else 0
        self.alphas[action] += reward
        self.betas[action] += 1 - reward
