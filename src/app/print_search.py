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
from metric import CumulativeInteractionMetricsEvaluator
from utils.dataset import Dataset
from utils.PersistentDataManager import PersistentDataManager
from utils.InteractorCache import InteractorCache
import metric
import matplotlib.pyplot as plt
from utils.DirectoryDependent import DirectoryDependent
from cycler import cycler
from collections import defaultdict

metrics_classes = [metric.Hits]
metrics_names = ['Cumulative Hits']

dm = DatasetManager()
datasets_preprocessors = dm.request_datasets_preprocessors()

interactors_search_parameters = yaml.load(
    open("settings" + sep + "interactors_search_parameters.yaml"),
    Loader=yaml.SafeLoader)

interactors_preprocessor_paramaters = yaml.load(
    open("settings" + sep + "interactors_preprocessor_parameters.yaml"),
    Loader=yaml.SafeLoader)
interactors_general_settings = yaml.load(
    open("settings" + sep + "interactors_general_settings.yaml"),
    Loader=yaml.SafeLoader)

evaluation_policies_parameters = yaml.load(
    open("settings" + sep + "evaluation_policies_parameters.yaml"),
    Loader=yaml.SafeLoader)

interactors_classes_names_to_names = {
    k: v['name'] for k, v in interactors_general_settings.items()
}

ir = InteractorRunner(dm, interactors_general_settings,
                      interactors_preprocessor_paramaters,
                      evaluation_policies_parameters)
interactors_classes = ir.select_interactors()

metrics_evaluator = CumulativeInteractionMetricsEvaluator(None, metrics_classes)

evaluation_policy = ir.get_interactors_evaluation_policy()

datasets_metrics_values = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))

for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)

    for metric_class_name in map(lambda x: x.__name__, metrics_classes):
        for itr_class in interactors_classes:
            for parameters in interactors_search_parameters[itr_class.__name__]:
                itr = itr_class(**parameters)
                pdm = PersistentDataManager(directory='results')

                metrics_pdm = PersistentDataManager(directory='metrics')
                metric_values = metrics_pdm.load(
                    os.path.join(
                        InteractorCache().get_id(dm, evaluation_policy, itr),
                        metrics_evaluator.get_id(), metric_class_name))
                datasets_metrics_values[dataset_preprocessor['name']][
                    metric_class_name][itr_class.__name__][','.join(map(str,list(parameters.values())))] = metric_values[-1]


for k1, v1 in datasets_metrics_values.items():
    for k2, v2 in v1.items():
        for k3, v3 in v2.items():
            values = np.array(list(v3.values()))
            keys = list(v3.keys())
            idxs = np.argsort(values)[::-1]
            keys = [keys[i] for i in idxs]
            values = [values[i] for i in idxs]
            
            for k4, v4 in zip(keys,values):
                print(f"{k3}({k4}) {v4:.5f}")
