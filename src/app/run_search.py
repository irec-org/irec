from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import interactors
import mf
from lib.utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from lib.utils.DatasetManager import DatasetManager
import yaml
from concurrent.futures import ProcessPoolExecutor
import argparse

parser = argparse.ArgumentParser(description='Grid search')

parser.add_argument('--num_tasks', type=int, default=os.cpu_count())
parser.add_argument('--forced_run', default=False, action='store_true')
parser.add_argument('-m',nargs='*')
parser.add_argument('-b',nargs='*')
args = parser.parse_args()
print(args.num_tasks)


def run_interactors_in_base(dataset_preprocessor, interactors_general_settings,
                            interactors_preprocessor_paramaters,
                            evaluation_policies_parameters, interactors_classes,
                            interactors_search_parameters):
    dm = DatasetManager()
    dm.initialize_engines(dataset_preprocessor)
    dm.load()
    # print(dm.dataset_preprocessed[0])
    # print(dm.dataset_preprocessed[1])
    ir = InteractorRunner(dm, interactors_general_settings,
                          interactors_preprocessor_paramaters,
                          evaluation_policies_parameters)
    ir.run_interactors_search(interactors_classes,
                              interactors_search_parameters, num_tasks= args.num_tasks,
                              forced_run=args.forced_run)


def main():

    interactors_preprocessor_paramaters = yaml.load(
        open("settings" + sep + "interactors_preprocessor_parameters.yaml"),
        Loader=yaml.SafeLoader)

    interactors_search_parameters = yaml.load(
        open("settings" + sep + "interactors_search_parameters.yaml"),
        Loader=yaml.SafeLoader)

    interactors_general_settings = yaml.load(
        open("settings" + sep + "interactors_general_settings.yaml"),
        Loader=yaml.SafeLoader)

    evaluation_policies_parameters = yaml.load(
        open("settings" + sep + "evaluation_policies_parameters.yaml"),
        Loader=yaml.SafeLoader)

    with open("settings"+sep+"datasets_preprocessors_parameters.yaml") as f:
        loader = yaml.SafeLoader
        datasets_preprocessors = yaml.load(f,Loader=loader)

        datasets_preprocessors = {setting['name']: setting
                                  for setting in datasets_preprocessors}

    dm = DatasetManager()

    # if args.b == None:
        # datasets_preprocessors = dm.request_datasets_preprocessors()
    # else:
    datasets_preprocessors = [datasets_preprocessors[base] for base in args.b]
    ir = InteractorRunner(None, interactors_general_settings,
                          interactors_preprocessor_paramaters,
                          evaluation_policies_parameters)
    # if args.m == None:
        # interactors_classes = ir.select_interactors()
    # else:
    interactors_classes = [eval('interactors.'+interactor) for interactor in args.m]
    print(interactors_classes)
    for dataset_preprocessor in datasets_preprocessors:
        run_interactors_in_base(dataset_preprocessor,
                            interactors_general_settings,
                            interactors_preprocessor_paramaters,
                            evaluation_policies_parameters,
                            interactors_classes,
                            interactors_search_parameters)


if __name__ == '__main__':
    main()
