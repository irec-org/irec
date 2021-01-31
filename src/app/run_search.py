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


def main():
    dm = DatasetManager()
    dataset_preprocessor = dm.request_dataset_preprocessor()
    dm.initialize_engines(dataset_preprocessor)
    dm.load()

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

    ir = InteractorRunner(dm, interactors_general_settings,
                          interactors_preprocessor_paramaters,
                          evaluation_policies_parameters
                          )
    ir.select_interactors()
    ir.run_interactors_search(interactors_search_parameters)

if __name__ == '__main__':
    main()
