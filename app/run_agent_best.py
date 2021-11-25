#!/usr/bin/python3

from os.path import dirname, realpath
import yaml
import os
import argparse

from irec.app import utils
import argparse

settings = utils.load_settings(dirname(realpath(__file__)))
parser = argparse.ArgumentParser()

parser.add_argument(
    "--evaluation_policy", default=settings["defaults"]["evaluation_policy"]
)
parser.add_argument(
    "--dataset_loaders", nargs="*", default=[settings["defaults"]["dataset_loader"]]
)
parser.add_argument("--agents", nargs="*", default=[settings["defaults"]["agent"]])

parser.add_argument("--tasks", type=int, default=os.cpu_count())

parser.add_argument("--forced_run", action='store_true', default=False)
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

settings["defaults"]["evaluation_policy"] = args.evaluation_policy

dataset_agents_parameters = yaml.load(
    open("./settings/dataset_agents.yaml"), Loader=yaml.SafeLoader
)

utils.run_agent_with_dataset_parameters(args.agents,args.dataset_loaders,settings,dataset_agents_parameters, args.tasks,args.forced_run)

