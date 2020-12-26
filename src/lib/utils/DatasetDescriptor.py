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

class DatasetDescriptor:
    def __init__(self,name=None,base_dir=None):
        name = name
        base_dir = base_dir
