#!/usr/bin/python3

import os
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
from app.constants import WORKINGDIR
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from app import utils

# from app import utils
import yaml
import subprocess
import argparse
import copy

settings = utils.load_settings()
parser = argparse.ArgumentParser()

parser.add_argument(
    "--evaluation_policy", default=settings["defaults"]["evaluation_policy"]
)
parser.add_argument(
    "--dataset_loaders", nargs="*", default=[settings["defaults"]["dataset_loader"]]
)
parser.add_argument("--agents", nargs="*", default=[settings["defaults"]["agent"]])

parser.add_argument("--tasks", type=int, default=os.cpu_count())

args = parser.parse_args()

settings["defaults"]["evaluation_policy"] = args.evaluation_policy


dataset_agents = yaml.load(
    open("./settings/dataset_agents.yaml"), Loader=yaml.SafeLoader
)

with ProcessPoolExecutor(max_workers=args.tasks) as executor:
    futures = set()
    for agent_name in args.agents:
        for dataset_loader_name in args.dataset_loaders:
            current_settings = copy.deepcopy(settings)
            current_settings["defaults"]["agent"] = agent_name
            current_settings["defaults"]["dataset_loader"] = dataset_loader_name
            current_settings["agents"][agent_name] = dataset_agents[
                dataset_loader_name
            ][agent_name]
            f = executor.submit(utils.run_agent, current_settings)
            futures.add(f)
            if len(futures) >= args.tasks:
                completed, futures = wait(futures, return_when=FIRST_COMPLETED)
    for f in futures:
        f.result()