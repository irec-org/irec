from .ICF import ICF
import numpy as np
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import ctypes
from collections import defaultdict
import joblib
import scipy
import mf
from utils.PersistentDataManager import PersistentDataManager
from .LinearICF import LinearICF
from .LinearUCB import LinearUCB


class LinearUCB1(LinearUCB):

    def train(self, train_dataset):
        super().train(train_dataset)
        self.bs = defaultdict(lambda: np.ones(self.num_latent_factors))
