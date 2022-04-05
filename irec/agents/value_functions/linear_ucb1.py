from .icf import ICF
import numpy as np
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import ctypes
from collections import defaultdict
import scipy
from . import mf
from .linear_icf import LinearICF
from .linear_ucb import LinearUCB


class LinearUCB1(LinearUCB):
    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.bs = defaultdict(lambda: np.ones(self.num_latent_factors))
