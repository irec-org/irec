from os.path import dirname, realpath, sep, pardir
import sys, os
sys.path.append(dirname(realpath(__file__)) + sep + pardir)

import numpy as np
import scipy.sparse

# from .. import lib.utils
from lib.utils.Parameterizable import Parameterizable
from collections import defaultdict
import pickle
import json
from .ValueFunction import ValueFunction

class ExperimentalValueFunction(ValueFunction,Parameterizable):
    def __init__(self,*args, **kwargs):
        ValueFunction.__init__(self,*args, **kwargs)
        Parameterizable.__init__(self)
    def reset(self,observation):
        train_dataset=observation
        ValueFunction.reset(self,train_dataset)
        self.t = 0
    def increment_time(self):
        self.t += 1
