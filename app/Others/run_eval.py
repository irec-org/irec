#!/usr/bin/python3
import sys
from os.path import dirname, pardir, realpath, sep
import copy

sys.path.append(dirname(realpath(__file__)) + sep + pardir)

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from app import utils
import irec.metrics

# from irec.utils.InteractorCache import InteractorCache
import numpy as np
import yaml

import irec.evaluation_policies
import irec.mf
import irec.value_functions
from app import constants
from irec.utils.dataset import Dataset
import argparse
import pickle

import mlflow
from mlflow.tracking import MlflowClient

settings = utils.load_settings()
parser = argparse.ArgumentParser()
# parser.add_argument("-i", default=[5, 10, 20, 50, 100], nargs="*")
parser.add_argument(
    "--evaluation_policy", default=settings["defaults"]["evaluation_policy"]
)
parser.add_argument("--dataset_loader", default=settings["defaults"]["dataset_loader"])
parser.add_argument("--agent", default=settings["defaults"]["agent"])
parser.add_argument("--metric", default=settings["defaults"]["metric"])
parser.add_argument(
    "--metric_evaluator", default=settings["defaults"]["metric_evaluator"]
)
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

# agent_name = args.agent
# dataset_name = args.dataset_loader
# evaluation_policy_name = args.evaluation_policy
# metric_name = args.metric
# metric_evaluator_name = args.metric_evaluator

settings["defaults"]["evaluation_policy"] = args.evaluation_policy
settings["defaults"]["agent"] = args.agent
settings["defaults"]["dataset_loader"] = args.dataset_loader
settings["defaults"]["metric"] = args.metric
settings["defaults"]["metric_evaluator"] = args.metric_evaluator


traintest_dataset = utils.load_dataset_experiment(settings)

data = np.vstack((traintest_dataset.train.data, traintest_dataset.test.data))

dataset = Dataset(data)
dataset.update_from_data()
dataset.update_num_total_users_items()

utils.evaluate_itr(dataset, settings)
