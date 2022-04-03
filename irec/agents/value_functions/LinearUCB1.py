from .ICF import ICF
import numpy as np
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import ctypes
from collections import defaultdict
import scipy
import mf
from .LinearICF import LinearICF
from .LinearUCB import LinearUCB


class LinearUCB1(LinearUCB):
    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.bs = defaultdict(lambda: np.ones(self.num_latent_factors))
