import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import scipy.sparse

from util import Saveable
from collections import defaultdict
import pickle
import json
from .Interactor import Interactor

class ExperimentalInteractor(Saveable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = dict()
    def train(self,train_data):
        super().train(train_data)
        self.t = 0
    def increment_time(self):
        self.t += 1
