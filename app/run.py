import pickle
from os.path import dirname, realpath, sep, pardir
import os
import sys

from mlflow.tracking.client import MlflowClient
from traitlets.traitlets import default

sys.path.append(dirname(dirname(realpath(__file__))))
import mlflow
from app import constants
import mlflow.tracking
import yaml
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from app import utils

import irec.value_functions
import irec.evaluation_policies
import argparse
import time


settings = utils.load_settings()
parser = argparse.ArgumentParser()
parser.add_argument("--evaluation_policy")
parser.add_argument("--dataset_loader")
parser.add_argument("--agent")

# for setting_name in ["defaults", "evaluation_policy", "agents", "dataset"]:
# for section_name, section_parameters in settings[setting_name].items():
# section_group = parser.add_argument_group(section_name)
# section_parameters = utils.flatten_dict(section_parameters)
# for k, v in section_parameters.items():
# section_group.add_argument(
# f"--{section_name}.{k}",
# default=v,
# type=type(v),
# )

utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

agent_name = args.agent
dataset_name = args.dataset_loader
evaluation_policy_name = args.evaluation_policy

dataset_loader_settings = settings["dataset_loaders"][dataset_name]

evaluation_policy_parameters = settings["evaluation_policies"][evaluation_policy_name]
evaluation_policy = eval("irec.evaluation_policies." + evaluation_policy_name)(
    **evaluation_policy_parameters
)

mlflow.set_experiment("dataset")

print(
    utils.parameters_normalize(
        constants.DATASET_PARAMETERS_PREFIX, dataset_name, dataset_loader_settings
    )
)
run = utils.already_ran(
    utils.parameters_normalize(
        constants.DATASET_PARAMETERS_PREFIX, dataset_name, dataset_loader_settings
    ),
    mlflow.get_experiment_by_name("dataset").experiment_id,
)

client = MlflowClient()
artifact_path = client.download_artifacts(run.info.run_id, "dataset.pickle")
traintest_dataset = pickle.load(open(artifact_path, "rb"))
print(traintest_dataset)

agent_parameters = settings["agents"][agent_name]
agent = utils.create_agent(agent_name, agent_parameters)
# agent_id = utils.get_agent_id(agent_name, parameters)
utils.run_interactor(
    agent=agent,
    agent_name=agent_name,
    agent_parameters=agent_parameters,
    dataset_name=dataset_name,
    dataset_parameters=dataset_loader_settings,
    traintest_dataset=traintest_dataset,
    evaluation_policy=evaluation_policy,
    evaluation_policy_name=evaluation_policy_name,
    evaluation_policy_parameters=evaluation_policy_parameters,
)
