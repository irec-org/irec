from .ICF import *
from .LinEGreedy import *
import numpy as np
import random
from tqdm import tqdm
#import util
from threadpoolctl import threadpool_limits
import ctypes
class LinearEGreedy(ICF,LinEGreedy):
    def __init__(self, *args, **kwargs):
        ICF.__init__(self, *args, **kwargs)
        LinEGreedy.__init__(self, *args, **kwargs)
    
    def init_A(self,num_lat):
        return self.user_lambda*np.eye(num_lat)
