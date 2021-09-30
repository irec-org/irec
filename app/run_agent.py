#!/usr/bin/python3
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
parser = argparse.ArgumentParser(add_help=False)

parser.add_argument(
    "--evaluation_policy", default=settings["defaults"]["evaluation_policy"]
)
parser.add_argument("--dataset_loader", default=settings["defaults"]["dataset_loader"])
parser.add_argument("--agent", default=settings["defaults"]["agent"])

utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

settings["defaults"]["evaluation_policy"] = args.evaluation_policy
settings["defaults"]["agent"] = args.agent
settings["defaults"]["dataset_loader"] = args.dataset_loader

data = utils.load_dataset_experiment(settings)
utils.run_agent(data, settings,True)
