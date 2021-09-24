import sys
from os.path import dirname, pardir, realpath, sep
import copy

sys.path.append(dirname(realpath(__file__)) + sep + pardir)

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from app import utils
import irec.metrics
import inquirer

# from irec.utils.InteractorCache import InteractorCache
import metrics
import numpy as np
import scipy.sparse
import yaml
from irec.metric_evaluators import (
    CumulativeInteractionMetricsEvaluator,
    CumulativeMetricsEvaluator,
    InteractionMetricsEvaluator,
    UserCumulativeInteractionMetricsEvaluator,
)
from sklearn.decomposition import NMF

import irec.evaluation_policies
import irec.mf
import irec.value_functions
from app import constants
from irec.utils.dataset import Dataset
import argparse
import pickle

import mlflow
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser()
# parser.add_argument("-i", default=[5, 10, 20, 50, 100], nargs="*")
parser.add_argument("--evaluation_policy")
parser.add_argument("--dataset_loader")
parser.add_argument("--agent")
parser.add_argument("--metric")
parser.add_argument("--metric_evaluator")
settings = utils.load_settings()
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()

settings = utils.sync_settings_from_args(settings, args)

agent_name = args.agent
dataset_name = args.dataset_loader
evaluation_policy_name = args.evaluation_policy
metric_name = args.metric
metric_evaluator_name = args.metric_evaluator

agent_parameters = settings["agents"][agent_name]

dataset_parameters = settings["dataset_loaders"][dataset_name]

evaluation_policy_name = settings["defaults"]["evaluation_policy"]
evaluation_policy_parameters = settings["evaluation_policies"][evaluation_policy_name]
evaluation_policy = eval("irec.evaluation_policies." + evaluation_policy_name)(
    **evaluation_policy_parameters
)

metric_evaluator_name = settings["defaults"]["metric_evaluator"]
metric_evaluator_parameters = settings["metric_evaluators"][metric_evaluator_name]


def evaluate_itr(
    agent_name,
    agent_parameters,
    dataset_name,
    dataset_parameters,
    evaluation_policy,
    evaluation_policy_name,
    evaluation_policy_parameters,
    metric_evaluator,
    metric_evaluator_name,
    metric_class,
    metric_class_name,
):
    mlflow.set_experiment("agent")
    parameters_agent_run = (
        utils.parameters_normalize(
            constants.DATASET_PARAMETERS_PREFIX, dataset_name, dataset_parameters
        )
        | utils.parameters_normalize(
            constants.EVALUATION_POLICY_PARAMETERS_PREFIX,
            evaluation_policy_name,
            evaluation_policy_parameters,
        )
        | utils.parameters_normalize(
            constants.AGENT_PARAMETERS_PREFIX, agent_name, agent_parameters
        )
    )
    # print(parameters_agent_run)
    run = utils.already_ran(
        parameters_agent_run, mlflow.get_experiment_by_name("agent").experiment_id
    )

    # run = utils.already_ran({'dataset': dataset_name}|,
    # mlflow.get_experiment_by_name('dataset').experiment_id)
    client = MlflowClient()
    artifact_path = client.download_artifacts(run.info.run_id, "interactions.pickle")
    interactions = pickle.load(open(artifact_path, "rb"))
    users_items_recommended = interactions

    if isinstance(metric_evaluator, CumulativeInteractionMetricsEvaluator):
        metric_values = metric_evaluator.evaluate(
            metric_class,
            users_items_recommended,
        )
    elif isinstance(metric_evaluator, InteractionMetricsEvaluator):
        metric_values = metric_evaluator.evaluate(
            metric_class,
            users_items_recommended,
        )
    elif isinstance(metric_evaluator, CumulativeMetricsEvaluator):
        metric_values = metric_evaluator.evaluate(metric_class, users_items_recommended)
    # metric_name = metric_class.__name__
    # metric_values
    mlflow.set_experiment("evaluation")
    parameters_evaluation_run = copy.copy(parameters_agent_run)
    parameters_evaluation_run |= utils.parameters_normalize(
        constants.METRIC_EVALUATOR_PARAMETERS_PREFIX, metric_evaluator_name, {}
    )
    parameters_evaluation_run |= utils.parameters_normalize(
        constants.METRIC_PARAMETERS_PREFIX, metric_class_name, {}
    )
    with mlflow.start_run() as run:
        utils.log_custom_parameters(parameters_evaluation_run)
        # print(metric_values)
        utils.log_custom_artifact("evaluation.pickle", metric_values)


metrics_classes = [metrics.Recall, metrics.Hits]

mlflow.set_experiment("dataset")
run = utils.already_ran(
    {"dataset": dataset_name}, mlflow.get_experiment_by_name("dataset").experiment_id
)

client = MlflowClient()
artifact_path = client.download_artifacts(run.info.run_id, "dataset.pickle")
traintest_dataset = pickle.load(open(artifact_path, "rb"))

data = np.vstack((traintest_dataset.train.data, traintest_dataset.test.data))

dataset = Dataset(data)
dataset.update_from_data()
dataset.update_num_total_users_items()

# metric_evaluator = InteractionMetricsEvaluator(ground_truth_dataset=dataset)

metric_class = eval("irec.metrics." + metric_name)

metric_evaluator = eval("irec.metric_evaluators." + metric_evaluator_name)(
    dataset, **metric_evaluator_parameters
)

evaluate_itr(
    agent_name=agent_name,
    agent_parameters=agent_parameters,
    dataset_name=dataset_name,
    dataset_parameters=dataset_parameters,
    evaluation_policy=evaluation_policy,
    evaluation_policy_name=evaluation_policy_name,
    evaluation_policy_parameters=evaluation_policy_parameters,
    metric_evaluator=metric_evaluator,
    metric_evaluator_name=metric_evaluator.__class__.__name__,
    metric_class=metric_class,
    metric_class_name=metric_class.__name__,
)
