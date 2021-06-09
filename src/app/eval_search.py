from os.path import dirname, realpath, sep, pardir
import os
import argparse
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir)

import inquirer
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import lib.value_functions
import lib.mf
import lib.evaluation_policies
from lib.utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import traceback
import scipy.sparse
from lib.utils.DatasetManager import DatasetManager
import yaml
from metrics import InteractionMetricsEvaluator, CumulativeMetricsEvaluator, CumulativeInteractionMetricsEvaluator
from lib.utils.dataset import Dataset
from lib.utils.PersistentDataManager import PersistentDataManager
import metrics
from lib.utils.utils import run_parallel
import utils
import ctypes
from copy import copy

parser = argparse.ArgumentParser(description='Grid search')

parser.add_argument('--num_tasks', type=int, default=os.cpu_count())
parser.add_argument('--forced_run', default=False, action='store_true')
parser.add_argument('-m',nargs='*')
parser.add_argument('-b',nargs='*')
settings = utils.load_settings()
utils.load_settings_to_parser(settings,parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings,args)

BUFFER_SIZE_EVALUATOR = 50

metrics_classes = [metrics.Hits]

evaluation_policy_name = settings['defaults']['interactors_evaluation_policy']
evaluation_policy_parameters = settings['evaluation_policies_parameters'][evaluation_policy_name]
evaluation_policy=eval('lib.evaluation_policies.'+evaluation_policy_name)(**evaluation_policy_parameters)

datasets_preprocessors = [settings['datasets_preprocessors_parameters'][base] for base in args.b]

for dataset_preprocessor in datasets_preprocessors:
    dm = DatasetManager()
    # dataset_preprocessor = dm.request_dataset_preprocessor()
    dm.initialize_engines(dataset_preprocessor)
    dm.load()

    data = np.vstack(
        (dm.dataset_preprocessed[0].data, dm.dataset_preprocessed[1].data))

    dataset = copy(dm.dataset_preprocessed[0])
    dataset.data = data
    dataset.update_from_data()

    metrics_evaluators = [
        CumulativeInteractionMetricsEvaluator(dataset, metrics_classes),
    ]

    def evaluate_itr(metric_evaluator_id, dm_id, agent, parameters):
        try:
            metric_evaluator = ctypes.cast(metric_evaluator_id,
                                           ctypes.py_object).value
            dm = ctypes.cast(dm_id, ctypes.py_object).value
            print(f"Evaluating {agent.name} results")
            pdm = PersistentDataManager(directory='results')

            metrics_pdm = PersistentDataManager(directory='metrics')

            users_items_recommended = pdm.load(utils.get_experiment_run_id(
                dm, evaluation_policy, agent))

            if isinstance(metric_evaluator, InteractionMetricsEvaluator):
                metrics_values = metric_evaluator.evaluate(
                    evaluation_policy.num_interactions,
                    evaluation_policy.interaction_size, users_items_recommended)
            elif isinstance(metric_evaluator, CumulativeMetricsEvaluator):
                metrics_values = metric_evaluator.evaluate(users_items_recommended)

            for metric_name, metric_values in metrics_values.items():
                if not args.forced_run and metrics_pdm.file_exists(
                        os.path.join(
                            utils.get_experiment_run_id(
                dm, evaluation_policy, agent),
                            metric_evaluator.get_id(), metric_name)):
                    raise SystemError
                metrics_pdm.save(
                    os.path.join(
                        utils.get_experiment_run_id(
                dm, evaluation_policy, agent),
                        metric_evaluator.get_id(), metric_name), metric_values)
        except Exception as e:
            traceback.print_exc()
            print(f"{e} ||| Error in evaluation, could not evaluate")


    with ProcessPoolExecutor() as executor:
        futures = set()
        for metric_evaluator in metrics_evaluators:
            for agent_name in args.m:
                for parameters in settings['agents_search_parameters'][agent_name]:
                    agent = utils.create_agent(agent_name,parameters)
                    f = executor.submit(evaluate_itr, id(metric_evaluator), id(dm),
                                        agent, parameters)
                    futures.add(f)
            if len(futures) >= args.num_tasks:
                completed, futures = wait(futures, return_when=FIRST_COMPLETED)
        for future in futures:
            future.result()
