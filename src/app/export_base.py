from os.path import dirname, realpath, sep, pardir
import numpy as np
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir)

import inquirer
import lib.utils.dataset as dataset
import yaml
from lib.utils.DirectoryDependent import DirectoryDependent
import lib.utils.splitters as splitters
import lib.utils.utils as util
import pickle
import utils
import argparse
from lib.utils.DatasetManager import DatasetManager

parser = argparse.ArgumentParser()
parser.add_argument('-b', nargs='*')
args = parser.parse_args()

settings = utils.load_settings()
dm = DatasetManager()
# datasets_preprocessors = dm.request_datasets_preprocessors()
datasets_preprocessors = [
    settings['datasets_preprocessors_parameters'][base] for base in args.b
]

for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)
    dm.load()
    # np.savetxt(DirectoryDependent()['export']+'train.csv',dm.dataset_preprocessed[0])
    # print(dm.train_dataset)
    # print(dm.test_dataset)
    np.savetxt(DirectoryDependent().DIRS['export']+'/train_'+dm.get_id()+'.csv',dm.dataset_preprocessed[0].data)
    np.savetxt(DirectoryDependent().DIRS['export']+'/test_'+dm.get_id()+'.csv',dm.dataset_preprocessed[1].data)
    # dm.dataset_preprocessed[1])
    # dm.run_preprocessor()
    # dm.save()
