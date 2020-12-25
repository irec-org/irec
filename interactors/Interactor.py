import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import scipy.sparse

from util import Saveable
from collections import defaultdict
import pickle
import json

class Interactor(Saveable):
    def __init__(self,evaluation_policy,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation_policy = evaluation_policy
