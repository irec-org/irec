from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
import interactors
import mf
from utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from utils.DatasetManager import DatasetManager
import yaml
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import time

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--num_tasks', type=int, default=os.cpu_count())
parser.add_argument('--forced_run', default=False, action='store_true')
args = parser.parse_args()

def run_interactors_in_base(dataset_preprocessor, interactors_general_settings,
                            interactors_preprocessor_paramaters,
                            evaluation_policies_parameters,
                            interactors_classes):
    dm = DatasetManager()
    dm.initialize_engines(dataset_preprocessor)
    dm.load()
    ir = InteractorRunner(dm, interactors_general_settings,
                          interactors_preprocessor_paramaters,
                          evaluation_policies_parameters)
    ir.run_interactors(interactors_classes,forced_run=args.forced_run)


def main():
    interactors_preprocessor_paramaters = yaml.load(
        open("settings" + sep + "interactors_preprocessor_parameters.yaml"),
        Loader=yaml.SafeLoader)
    interactors_general_settings = yaml.load(
        open("settings" + sep + "interactors_general_settings.yaml"),
        Loader=yaml.SafeLoader)

    evaluation_policies_parameters = yaml.load(
        open("settings" + sep + "evaluation_policies_parameters.yaml"),
        Loader=yaml.SafeLoader)
    dm = DatasetManager()
    datasets_preprocessors = dm.request_datasets_preprocessors()
    ir = InteractorRunner(None, interactors_general_settings,
                          interactors_preprocessor_paramaters,
                          evaluation_policies_parameters)
    interactors_classes = ir.select_interactors()
    with ProcessPoolExecutor() as executor:
        futures = set()
        for dataset_preprocessor in datasets_preprocessors:
            f = executor.submit(run_interactors_in_base, dataset_preprocessor,
                                interactors_general_settings,
                                interactors_preprocessor_paramaters,
                                evaluation_policies_parameters,
                                interactors_classes)
            futures.add(f)

            if len(futures) >= args.num_tasks:
                completed, futures = wait(futures, return_when=FIRST_COMPLETED)

        for f in futures:
            f.result()


if __name__ == '__main__':
    main()
