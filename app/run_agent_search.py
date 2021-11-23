#!/usr/bin/python3

from os.path import dirname, realpath, sep, pardir
import yaml
import sys
import os
import argparse

sys.path.append(dirname(realpath(__file__)) + sep + pardir)
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

agents_search = yaml.load(open("./settings/agents_search.yaml"), Loader=yaml.SafeLoader)

settings["defaults"]["evaluation_policy"] = args.evaluation_policy


utils.run_agent_search_with_dataset_parameters(args.agents,args.dataset_loaders,
        settings,agents_search, args.tasks,args.forced_run)
