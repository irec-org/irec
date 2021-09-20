import sys
from copy import copy
from os.path import dirname, pardir, realpath, sep

sys.path.append(dirname(realpath(__file__)) + sep + pardir)

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from app import utils
import lib.metrics
import inquirer
# from lib.utils.InteractorCache import InteractorCache
import metrics
import numpy as np
import scipy.sparse
import yaml
from metrics import (CumulativeInteractionMetricsEvaluator,
        CumulativeMetricsEvaluator, InteractionMetricsEvaluator,
        UserCumulativeInteractionMetricsEvaluator)
from sklearn.decomposition import NMF

import lib.evaluation_policies
import lib.mf
import lib.value_functions
from app import constants
from lib.utils.dataset import Dataset
import argparse
import pickle

import mlflow
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument('-m')
parser.add_argument('-b')
parser.add_argument('-e')
parser.add_argument('--forced_run', default=False, action='store_true')
parser.add_argument('-i', default=[5, 10, 20, 50, 100], nargs='*')
settings = utils.load_settings()
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

agent_name = args.m
dataset_name = args.b
forced_run = args.forced_run

agent_parameters = settings['agents_preprocessor_parameters'][dataset_name][
        agent_name]

dataset_parameters = settings['dataset_loaders'][dataset_name]

evaluation_policy_name = settings['defaults']['interactors_evaluation_policy']
evaluation_policy_parameters = settings['evaluation_policies_parameters'][
        evaluation_policy_name]
evaluation_policy = eval('lib.evaluation_policies.' + evaluation_policy_name)(
        **evaluation_policy_parameters)


def evaluate_itr(agent_name, agent_parameters, dataset_name,
        dataset_parameters, evaluation_policy, evaluation_policy_name,
        evaluation_policy_parameters, metric_evaluator,
        metric_evaluator_name, metric_class, metric_name,
        forced_run: bool):
    mlflow.set_experiment('agent')
    parameters_agent_run = utils.parameters_normalize(constants.DATASET_PARAMETERS_PREFIX, dataset_name,dataset_parameters) |\
            utils.parameters_normalize(constants.EVALUATION_POLICY_PARAMETERS_PREFIX, evaluation_policy_name, evaluation_policy_parameters)|\
            utils.parameters_normalize(constants.AGENT_PARAMETERS_PREFIX, agent_name, agent_parameters)
    run = utils.already_ran(
            parameters_agent_run,
            mlflow.get_experiment_by_name('agent').experiment_id)

    # run = utils.already_ran({'dataset': dataset_name}|,
    # mlflow.get_experiment_by_name('dataset').experiment_id)
    client = MlflowClient()
    artifact_path = client.download_artifacts(run.info.run_id,
            'interactions.pickle')
    interactions = pickle.load(open(artifact_path, 'rb'))
    users_items_recommended = interactions

    if isinstance(metric_evaluator, CumulativeInteractionMetricsEvaluator):
        metric_values = metric_evaluator.evaluate(
                metric_class,
                evaluation_policy.num_interactions,
                evaluation_policy.interaction_size,
                users_items_recommended,
                interactions_to_evaluate=nums_interactions_to_show)
    elif isinstance(metric_evaluator, InteractionMetricsEvaluator):
        metric_values = metric_evaluator.evaluate(
                metric_class, evaluation_policy.num_interactions,
                evaluation_policy.interaction_size, users_items_recommended)
    elif isinstance(metric_evaluator, CumulativeMetricsEvaluator):
        metric_values = metric_evaluator.evaluate(users_items_recommended)
    # metric_name = metric_class.__name__
    # metric_values
    mlflow.set_experiment('evaluation')
    parameters_evaluation_run = copy(parameters_agent_run)
    parameters_evaluation_run |= utils.parameters_normalize(
            constants.METRIC_EVALUATOR_PARAMETERS_PREFIX, metric_evaluator_name,
            {})
    parameters_evaluation_run |= utils.parameters_normalize(
            constants.METRIC_PARAMETERS_PREFIX, metric_name, {})
    with mlflow.start_run():
        utils.log_custom_parameters(parameters_evaluation_run)
        utils.log_custom_artifact('evaluation.pickle', metric_values)


BUFFER_SIZE_EVALUATOR = 50

nums_interactions_to_show = list(map(int, args.i))

metrics_classes = [metrics.Recall, metrics.Hits]

mlflow.set_experiment('dataset')
run = utils.already_ran({'dataset': dataset_name},
        mlflow.get_experiment_by_name('dataset').experiment_id)

client = MlflowClient()
artifact_path = client.download_artifacts(run.info.run_id, 'dataset.pickle')
traintest_dataset = pickle.load(open(artifact_path, 'rb'))

data = np.vstack((traintest_dataset.train.data, traintest_dataset.test.data))

dataset = Dataset(data)
dataset.update_from_data()
dataset.update_num_total_users_items()

metric_evaluator = UserCumulativeInteractionMetricsEvaluator(dataset)

metric_class = eval('lib.metrics.' + args.e)

evaluate_itr(agent_name=agent_name,
        agent_parameters=agent_parameters,
        dataset_name=dataset_name,
        dataset_parameters=dataset_parameters,
        evaluation_policy=evaluation_policy,
        evaluation_policy_name=evaluation_policy_name,
        evaluation_policy_parameters=evaluation_policy_parameters,
        metric_evaluator=metric_evaluator,
        metric_evaluator_name=metric_evaluator.__class__.__name__,
        metric_class=metric_class,
        metric_name=metric_class.__name__,
        forced_run=forced_run)
