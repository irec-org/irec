import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-m', nargs='*')
parser.add_argument('-b', nargs='*')
parser.add_argument('-r', nargs='*',type=float,default=[0.1,0.2,0.3,0.4,0.5])
# parser.add_argument('-r', nargs='*',type=float,default=[0.1,0.2])
parser.add_argument('--num_tasks', type=int, default=3)
settings = utils.load_settings()
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir)

import lib.evaluation_policies
import inquirer
import lib.value_functions
import lib.mf
from lib.utils.InteractorRunner import InteractorRunner
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from lib.utils.DatasetManager import DatasetManager
import yaml
from lib.metrics import InteractionMetricsEvaluator, CumulativeMetricsEvaluator, CumulativeInteractionMetricsEvaluator, UserCumulativeInteractionMetricsEvaluator
from lib.utils.dataset import Dataset
# from lib.utils.InteractorCache import InteractorCache
from lib.utils.utils import run_parallel
import lib.metrics
import ctypes


def evaluate_itr(dataset ,dm_id, agent_name,evaluation_policy):
    dm = ctypes.cast(dm_id, ctypes.py_object).value
    print(f"Evaluating {agent_name} results")
    metric_evaluator = CumulativeMetricsEvaluator(ground_truth_dataset=dataset, metrics_classes=metrics_classes,buffer_size=BUFFER_SIZE_EVALUATOR)
    parameters = settings['agents_preprocessor_parameters'][
        dm.dataset_preprocessor.name][agent_name]
    agent = utils.create_agent(agent_name, parameters)
    agent_id = utils.get_agent_id(agent_name, parameters)

    users_items_recommended = pdm.load(
        utils.get_experiment_run_id(dm, evaluation_policy, agent_id))

    metrics_values = metric_evaluator.evaluate(users_items_recommended)

    for metric_name, metric_values in metrics_values.items():
        metrics_pdm.save(
            os.path.join(
                utils.get_experiment_run_id(dm, evaluation_policy, agent_id),
                metric_evaluator.get_id(), metric_name), metric_values)


# parser = argparse.ArgumentParser(description='Grid search')
# BUFFER_SIZE_EVALUATOR = 1000
BUFFER_SIZE_EVALUATOR = 100000

metrics_classes = [lib.metrics.Recall, lib.metrics.Hits, lib.metrics.NumInteractions]
# metrics_classes = [lib.metrics.NumInteractions]

datasets_preprocessors = [
    settings['datasets_preprocessors_parameters'][base] for base in args.b
]

dm = DatasetManager()

ir = InteractorRunner(dm, settings['interactors_general_settings'],
                      settings['agents_preprocessor_parameters'],
                      settings['evaluation_policies_parameters'])
with ProcessPoolExecutor() as executor:
    futures = set()
    for dataset_preprocessor in datasets_preprocessors:

        dm.initialize_engines(dataset_preprocessor)
        dm.load()

        ir = InteractorRunner(dm, settings['interactors_general_settings'],
                              settings['agents_preprocessor_parameters'],
                              settings['evaluation_policies_parameters'])

        data = np.vstack(
            (dm.dataset_preprocessed[0].data, dm.dataset_preprocessed[1].data))

        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        for hit_rate in args.r:
            evaluation_policy_name = "LimitedInteraction"
            evaluation_policy_parameters = settings['evaluation_policies_parameters'][
                evaluation_policy_name]
            evaluation_policy = eval('lib.evaluation_policies.' +
                                     evaluation_policy_name)(
                                         **evaluation_policy_parameters)
            evaluation_policy.recommend_test_data_rate_limit = hit_rate

            for agent_name in args.m:
                f = executor.submit(evaluate_itr,dataset, id(dm), agent_name,evaluation_policy)
                futures.add(f)
                if len(futures) >= args.num_tasks:
                    completed, futures = wait(futures,
                            return_when=FIRST_COMPLETED)
                    for f in futures:
                        f.result()
    for f in futures:
        f.result()
