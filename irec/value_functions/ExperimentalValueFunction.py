from os.path import dirname, realpath, sep, pardir
import sys, os
sys.path.append(dirname(realpath(__file__)) + sep + pardir)

import numpy as np
import scipy.sparse

# from .. import irec.utils
from collections import defaultdict
import pickle
import json
from .ValueFunction import ValueFunction


class ExperimentalValueFunction(ValueFunction):
    def __init__(self, *args, **kwargs):
        ValueFunction.__init__(self, *args, **kwargs)

    def reset(self, observation):
        train_dataset = observation
        ValueFunction.reset(self, train_dataset)
