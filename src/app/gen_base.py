from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
import utils.dataset as dataset
import yaml
from lib.DirectoryDependent import DirectoryDependent
import utils.splitters as splitters
import utils.util as util
import pickle
from lib.DatasetManager import DatasetManager
dm = DatasetManager()
dm.request_dataset_preprocessor()
dm.initialize_engines()
dm.run_parser()
dm.run_splitter()
dm.save()
