from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
import interactors
import mf
from utils.InteractorsRunner import InteractorsRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from lib.DatasetManager import DatasetManager
dm = DatasetManager()
dm.request_dataset_preprocessor()
dm.initialize_engines()

ir = InteractorsRunner(dm)
ir.select_interactors()
ir.run_interactors()
# ir.run_bases(['tr_te_yahoo_music',
#               'tr_te_good_books','tr_te_ml_10m'])
