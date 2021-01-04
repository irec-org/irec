import pandas as pd
import numpy as np
from collections import defaultdict
import random
import math
import time
import scipy.sparse
import os
import re
from Saveable import Saveable
import dataset_parsers
from dataclasses import dataclass

class DatasetDescriptor:
    def __init__(self,name=None,base_dir=None):
        name = name
        base_dir = base_dir

class Dataset:
    def __init__(self,data,num_users=None,num_items=None,rate_domain=None,uids=None):
        self.data = data
        self.num_users = num_users
        self.num_items = num_items
        self.rate_domain = rate_domain
        self.uids = uids

    def update_from_data(self):
        self.num_users = len(np.unique(self.data[0]))
        self.num_items = len(np.unique(self.data[1]))
        self.rate_domain  = set(np.unique(self.data[2]))
        self.uids = np.unique(self.data[0])

