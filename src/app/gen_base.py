from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
import lib.utils.dataset as dataset
import yaml
from lib.utils.DirectoryDependent import DirectoryDependent
import lib.utils.splitters as splitters
import lib.utils.utils as util
import pickle
from lib.utils.DatasetManager import DatasetManager
dm = DatasetManager()
datasets_preprocessors = dm.request_datasets_preprocessors()
for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)
    dm.run_preprocessor()
    dm.save()
