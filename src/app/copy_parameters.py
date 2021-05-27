from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir)

import json
import inquirer
import lib.interactors
import lib.mf
import utils
import lib.evaluation_policies
from lib.utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from lib.utils.DatasetManager import DatasetManager
import yaml
from metrics import CumulativeInteractionMetricsEvaluator
from lib.utils.dataset import Dataset
from lib.utils.PersistentDataManager import PersistentDataManager
from lib.utils.InteractorCache import InteractorCache
import metrics
import matplotlib.pyplot as plt
from lib.utils.DirectoryDependent import DirectoryDependent
from cycler import cycler
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='Grid search')
parser.add_argument('-m',nargs='*')
parser.add_argument('-s',nargs='*')
parser.add_argument('-t',nargs='*')
settings = utils.load_settings()
utils.load_settings_to_parser(settings,parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings,args)

for method in args.m:
    for source,target in zip(args.s,args.t):
        settings['interactors_preprocessor_parameters'][target][method] = settings['interactors_preprocessor_parameters'][source][method]

open("settings" + sep + "interactors_preprocessor_parameters.yaml",'w').write(yaml.dump(utils.default_to_regular(settings['interactors_preprocessor_parameters'])))
