import pickle
from os.path import dirname, realpath, sep, pardir
import os
import sys

from mlflow.tracking.client import MlflowClient
sys.path.append(dirname(dirname(realpath(__file__))))
import mlflow
from app import constants
import mlflow.tracking
import yaml
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from app import utils
import lib.value_functions
import lib.evaluation_policies
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--forced_run', default=False, action='store_true')

parser.add_argument('-m')
parser.add_argument('-b')
settings = utils.load_settings()
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

agent_name = args.m
dataset_name = args.b
forced_run = args.forced_run

dataset_loader_settings = settings['dataset_loaders'][dataset_name]

evaluation_policy_name = settings['defaults']['interactors_evaluation_policy']
evaluation_policy_parameters = settings['evaluation_policies_parameters'][
    evaluation_policy_name]
evaluation_policy = eval('lib.evaluation_policies.' + evaluation_policy_name)(
    **evaluation_policy_parameters)

mlflow.set_experiment('dataset')

print(utils.parameters_normalize(constants.DATASET_PARAMETERS_PREFIX, args.b,
                                   dataset_loader_settings))
run = utils.already_ran(
        utils.parameters_normalize(constants.DATASET_PARAMETERS_PREFIX, args.b,
                                   dataset_loader_settings),
    mlflow.get_experiment_by_name('dataset').experiment_id)

client = MlflowClient()
artifact_path = client.download_artifacts(run.info.run_id, 'dataset.pickle')
traintest_dataset = pickle.load(open(artifact_path, 'rb'))
print(traintest_dataset)

agent_parameters = settings['agents_preprocessor_parameters'][dataset_name][
    agent_name]
agent = utils.create_agent(agent_name, agent_parameters)
# agent_id = utils.get_agent_id(agent_name, parameters)
utils.run_interactor(agent=agent,
                     agent_name=agent_name,
                     agent_parameters=agent_parameters,
                     dataset_name=dataset_name,
                     dataset_parameters=dataset_loader_settings,
                     traintest_dataset=traintest_dataset,
                     evaluation_policy=evaluation_policy,
                     evaluation_policy_name=evaluation_policy_name,
                     evaluation_policy_parameters=evaluation_policy_parameters,
                     forced_run=forced_run)
