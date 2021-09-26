#!/usr/bin/python3

from os.path import dirname, realpath
import os
import sys

# sys.path.append(dirname(dirname(realpath(__file__))))
sys.path.append(dirname(dirname(realpath(__file__))))
# print(os.path.join(dirname(realpath(__file__)), "..", "app"))
# from utils import flatten_dict
from app import utils

# from app import utils
import yaml
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

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

utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

agents_search = yaml.load(open("./settings/agents_search.yaml"), Loader=yaml.SafeLoader)

settings["defaults"]["evaluation_policy"] = args.evaluation_policy
# settings["defaults"]["agent"] = args.agent
# settings["defaults"]["dataset_loader"] = args.dataset_loader

# evaluation_policy_parameters = settings["evaluation_policies"][evaluation_policy_name]
# agent_parameters = settings["agents"][agent_name]
# dataset_loader_parameters = settings["dataset_loaders"][dataset_loader_name]

# subsettings = {k: v for k, v in settings.items() if k not in ["agents"]}
with ProcessPoolExecutor(max_workers=args.tasks) as executor:
    futures = set()
    for dataset_loader_name in args.dataset_loaders:
        settings["defaults"]["dataset_loader"] = dataset_loader_name
        data = utils.load_dataset_experiment(settings)
        for agent_name in args.agents:
            settings["defaults"]["agent"] = agent_name
            for agent_og_parameters in agents_search[agent_name]:
                settings["agents"][agent_name] = agent_og_parameters
                # agent_og_parameters = dataset_agents[dataset_loader_name][agent_name]
                f = executor.submit(utils.run_agent, data, copy.deepcopy(settings))
                futures.add(f)
                if len(futures) >= args.tasks:
                    completed, futures = wait(futures, return_when=FIRST_COMPLETED)
    for f in futures:
        f.result()
