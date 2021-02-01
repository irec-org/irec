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
datasets_preprocessors = dm.request_datasets_preprocessors()
for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)
    dm.run_preprocessor()
    dm.save()
