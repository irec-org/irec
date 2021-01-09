import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import scipy.sparse

from util import Saveable
from collections import defaultdict
import pickle
import json

class Interactor:
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    def train(self,train_data):
        super().train(train_data)
    def predict(self,uid,candidate_items,num_req_items):
        return None, None
    def update(self,uid,item,reward,additional_data):
        pass
