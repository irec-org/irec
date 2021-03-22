from os.path import dirname, realpath, sep, pardir
import os
import argparse
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import interactors
import mf
from utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from utils.DatasetManager import DatasetManager
import yaml
from metric import InteractionMetricsEvaluator, CumulativeMetricsEvaluator, CumulativeInteractionMetricsEvaluator
from utils.dataset import Dataset
from utils.PersistentDataManager import PersistentDataManager
from utils.InteractorCache import InteractorCache
import metric
from utils.util import run_parallel
import ctypes
from copy import copy

parser = argparse.ArgumentParser(description='Grid search')

parser.add_argument('--num_tasks', type=int, default=os.cpu_count())
parser.add_argument('--forced_run', default=False, action='store_true')
parser.add_argument('-m',nargs='*')
parser.add_argument('-b',nargs='*')
args = parser.parse_args()

BUFFER_SIZE_EVALUATOR = 50

metrics_classes = [metric.Hits]


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

with open("settings"+sep+"datasets_preprocessors_parameters.yaml") as f:
    loader = yaml.SafeLoader
    datasets_preprocessors = yaml.load(f,Loader=loader)

    datasets_preprocessors = {setting['name']: setting
                              for setting in datasets_preprocessors}

datasets_preprocessors = [datasets_preprocessors[base] for base in args.b]

for dataset_preprocessor in datasets_preprocessors:
    dm = DatasetManager()
    # dataset_preprocessor = dm.request_dataset_preprocessor()
    dm.initialize_engines(dataset_preprocessor)
    dm.load()
    ir = InteractorRunner(dm, interactors_general_settings,
                          interactors_preprocessor_paramaters,
                          evaluation_policies_parameters)
    # interactors_classes = ir.select_interactors()
    interactors_classes = [eval('interactors.'+interactor) for interactor in args.m]

    data = np.vstack(
        (dm.dataset_preprocessed[0].data, dm.dataset_preprocessed[1].data))

    dataset = copy(dm.dataset_preprocessed[0])
    dataset.data = data
    # dataset = Dataset(data)
    dataset.update_from_data()
    # dataset.update_num_total_users_items()

    metrics_evaluators = [
        CumulativeInteractionMetricsEvaluator(dataset, metrics_classes),
    ]

    evaluation_policy = ir.get_interactors_evaluation_policy()


    def evaluate_itr(metric_evaluator_id, dm_id, itr_class, parameters):
        try:
            metric_evaluator = ctypes.cast(metric_evaluator_id,
                                           ctypes.py_object).value
            dm = ctypes.cast(dm_id, ctypes.py_object).value
            print(f"Evaluating {itr_class.__name__} results")
            itr = itr_class(**parameters)
            pdm = PersistentDataManager(directory='results')

            metrics_pdm = PersistentDataManager(directory='metrics')

            users_items_recommended = pdm.load(InteractorCache().get_id(
                dm, evaluation_policy, itr))

            if isinstance(metric_evaluator, InteractionMetricsEvaluator):
                metrics_values = metric_evaluator.evaluate(
                    evaluation_policy.num_interactions,
                    evaluation_policy.interaction_size, users_items_recommended)
            elif isinstance(metric_evaluator, CumulativeMetricsEvaluator):
                metrics_values = metric_evaluator.evaluate(users_items_recommended)

            for metric_name, metric_values in metrics_values.items():
                if not args.forced_run and metrics_pdm.file_exists(
                        os.path.join(
                            InteractorCache().get_id(dm, evaluation_policy, itr),
                            metric_evaluator.get_id(), metric_name)):
                    raise SystemError
                metrics_pdm.save(
                    os.path.join(
                        InteractorCache().get_id(dm, evaluation_policy, itr),
                        metric_evaluator.get_id(), metric_name), metric_values)
        except Exception as e:
            print(f"{e} ||| Error in evaluation, could not evaluate")


    with ProcessPoolExecutor() as executor:
        futures = set()
        for metric_evaluator in metrics_evaluators:
            for itr_class in interactors_classes:
                for parameters in interactors_search_parameters[itr_class.__name__]:
                    f = executor.submit(evaluate_itr, id(metric_evaluator), id(dm),
                                        itr_class, parameters)
                    futures.add(f)
            if len(futures) >= args.num_tasks:
                completed, futures = wait(futures, return_when=FIRST_COMPLETED)
        for future in futures:
            future.result()
