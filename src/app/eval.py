import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m',nargs='*')
parser.add_argument('-b',nargs='*')
args = parser.parse_args()

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
from metric import InteractionMetricsEvaluator, CumulativeMetricsEvaluator, CumulativeInteractionMetricsEvaluator, UserCumulativeInteractionMetricsEvaluator
from utils.dataset import Dataset
from utils.PersistentDataManager import PersistentDataManager
from utils.InteractorCache import InteractorCache
import metric
from utils.util import run_parallel
import ctypes


def evaluate_itr(metric_evaluator_id, dm_id, itr_class):
    metric_evaluator = ctypes.cast(metric_evaluator_id, ctypes.py_object).value
    dm = ctypes.cast(dm_id, ctypes.py_object).value
    print(f"Evaluating {itr_class.__name__} results")
    itr = ir.create_interactor(itr_class)
    pdm = PersistentDataManager(directory='results')
    users_items_recommended = pdm.load(InteractorCache().get_id(
        dm, evaluation_policy, itr))

    metrics_pdm = PersistentDataManager(directory='metrics')
    if isinstance(metric_evaluator, CumulativeInteractionMetricsEvaluator):
        metrics_values = metric_evaluator.evaluate(
            evaluation_policy.num_interactions,
            evaluation_policy.interaction_size, users_items_recommended,interactions_to_evaluate=nums_interactions_to_show)
    elif isinstance(metric_evaluator, InteractionMetricsEvaluator):
        metrics_values = metric_evaluator.evaluate(
            evaluation_policy.num_interactions,
            evaluation_policy.interaction_size, users_items_recommended)
    elif isinstance(metric_evaluator, CumulativeMetricsEvaluator):
        metrics_values = metric_evaluator.evaluate(users_items_recommended)

    for metric_name, metric_values in metrics_values.items():
        metrics_pdm.save(
            os.path.join(InteractorCache().get_id(dm, evaluation_policy, itr),
                         metric_evaluator.get_id(), metric_name), metric_values)


parser = argparse.ArgumentParser(description='Grid search')
BUFFER_SIZE_EVALUATOR = 50

nums_interactions_to_show = [5, 10, 20, 50, 100]

metrics_classes = [metric.Recall, metric.Hits, metric.EPC, metric.UsersCoverage, metric.ILD,metric.GiniCoefficientInv]
# metrics_classes = [metric.Recall, metric.Hits, metric.EPC, metric.UsersCoverage, metric.ILD]
# metrics_classes = [metric.GiniCoefficientInv]
#metrics_classes = [metric.Recall, metric.Hits, metric.EPC]
# metrics_classes = [metric.ILD,metric.UsersCoverage]
# metrics_classes = [metric.ILD]

interactors_preprocessor_paramaters = yaml.load(
    open("settings" + sep + "interactors_preprocessor_parameters.yaml"),
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
datasets_preprocessors = [datasets_preprocessors[base] for base in args.b]

dm = DatasetManager()

ir = InteractorRunner(dm, interactors_general_settings,
                      interactors_preprocessor_paramaters,
                      evaluation_policies_parameters)
interactors_classes = [eval('interactors.'+interactor) for interactor in args.m]
for dataset_preprocessor in datasets_preprocessors:
    
    dm.initialize_engines(dataset_preprocessor)
    dm.load()

    ir = InteractorRunner(dm, interactors_general_settings,
                          interactors_preprocessor_paramaters,
                          evaluation_policies_parameters)

    data = np.vstack(
        (dm.dataset_preprocessed[0].data, dm.dataset_preprocessed[1].data))

    dataset = Dataset(data)
    dataset.update_from_data()
    dataset.update_num_total_users_items()

    metrics_evaluators = [
        UserCumulativeInteractionMetricsEvaluator(dataset, metrics_classes)
    ]

    evaluation_policy = ir.get_interactors_evaluation_policy()

    for metric_evaluator in metrics_evaluators:
        # args = [(id(metric_evaluator), id(dm), itr_class)
                # for itr_class in interactors_classes]
        # run_parallel(evaluate_itr, args, use_tqdm=False)
        for itr_class in interactors_classes:
            evaluate_itr(id(metric_evaluator), id(dm), itr_class)
