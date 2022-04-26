import numpy as np
from collections import defaultdict
from .linear_ucb import LinearUCB


class LinearUCB1(LinearUCB):
    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.bs = defaultdict(lambda: np.ones(self.num_latent_factors))
