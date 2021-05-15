from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir)

import inquirer
import lib.interactors
import mf
from lib.utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from lib.utils.DatasetManager import DatasetManager
import lib.evaluation_policies
import yaml
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import time
import utils

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--num_tasks', type=int, default=os.cpu_count())
parser.add_argument('--forced_run', default=False, action='store_true')
parser.add_argument('--parallel', default=False, action='store_true')

parser.add_argument('-m',nargs='*')
parser.add_argument('-b',nargs='*')
settings = utils.load_settings()
utils.load_settings_to_parser(settings,parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings,args)

evaluation_policy_name = settings['defaults']['interactors_evaluation_policy']
evaluation_policy_parameters = settings['evaluation_policies_parameters'][evaluation_policy_name]
evaluation_policy=eval('lib.evaluation_policies.'+evaluation_policy_name)(**evaluation_policy_parameters)

def main():
    dm = DatasetManager()
    datasets_preprocessors = [settings['datasets_preprocessors_parameters'][base] for base in args.b]

    interactors_classes = [
        eval('lib.interactors.' + interactor) for interactor in args.m
    ]
        
    with ProcessPoolExecutor() as executor:
        futures = set()
        for dataset_preprocessor in datasets_preprocessors:
            dm = DatasetManager()
            dm.initialize_engines(dataset_preprocessor)
            dm.load()
            for itr_class in interactors_classes:
                itr = itr_class(**settings['interactors_preprocessor_paramaters'][dataset_preprocessor['name']][itr_class.__name__]['parameters'])
                f=executor.submit(utils.run_interactor,itr,evaluation_policy,dm,args.forced_run)
                futures.add(f)
                if len(futures) >= args.num_tasks:
                    completed, futures = wait(futures, return_when=FIRST_COMPLETED)
        for f in futures:
            f.result()
if __name__ == '__main__':
    main()
