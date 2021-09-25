#!/bin/python3

from os.path import dirname, realpath
import os
import sys

# sys.path.append(dirname(dirname(realpath(__file__))))
sys.path.append(dirname(dirname(realpath(__file__))))
# print(os.path.join(dirname(realpath(__file__)), "..", "app"))
# from utils import flatten_dict
from app import utils

# from app import utils
import subprocess
import yaml
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

settings = utils.load_settings()
parser = argparse.ArgumentParser()

parser.add_argument(
    "--evaluation_policy", default=settings["defaults"]["evaluation_policy"]
)
parser.add_argument("--dataset_loader", default=settings["defaults"]["dataset_loader"])
parser.add_argument("--agent", default=settings["defaults"]["agent"])

parser.add_argument("--tasks", type=int)
# utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
# settings = utils.sync_settings_from_args(settings, args)

agents_search = yaml.load(open("./settings/agents_search.yaml"), Loader=yaml.SafeLoader)
PYTHONCMD = "python3"
WORKINGDIR = "."
# dataset_agents
agent_name = args.agent
dataset_loader_name = args.dataset_loader
evaluation_policy_name = args.evaluation_policy

# evaluation_policy_parameters = settings["evaluation_policies"][evaluation_policy_name]
# agent_parameters = settings["agents"][agent_name]
# dataset_loader_parameters = settings["dataset_loaders"][dataset_loader_name]

# subsettings = {k: v for k, v in settings.items() if k not in ["agents"]}
with ProcessPoolExecutor(max_workers=args.tasks) as executor:
    futures = set()
    for agent_og_parameters in agents_search[agent_name]:
        # agent_og_parameters = dataset_agents[dataset_loader_name][agent_name]

        agent_parameters = utils.flatten_dict(
            {"agents": {agent_name: agent_og_parameters}}
        )
        agent_str_parameters = " ".join(
            ["--{}='{}'".format(k, v) for k, v in agent_parameters.items()]
        )
        f = executor.submit(
            subprocess.run,
            "{} ./run_agent.py --dataset_loader '{}' --agent '{}' --evaluation_policy '{}' {}".format(
                PYTHONCMD,
                dataset_loader_name,
                agent_name,
                evaluation_policy_name,
                agent_str_parameters,
            ),
            cwd=WORKINGDIR,
        )
        futures.add(f)
        if len(futures) >= args.tasks:
            completed, futures = wait(futures, return_when=FIRST_COMPLETED)
    for f in futures:
        f.result()
