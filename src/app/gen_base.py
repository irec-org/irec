from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
import utils.dataset as dataset
import yaml
from utils.DirectoryDependent import DirectoryDependent
import utils.splitters as splitters
import utils.util as util
import pickle
from utils.DatasetManager import DatasetManager
dm = DatasetManager()
dataset_preprocessor = dm.request_dataset_preprocessor()
dm.initialize_engines(dataset_preprocessor)
dm.run_preprocessor()
dm.save()
