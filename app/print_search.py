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
parser.add_argument("-t", default=False, action="store_true", help="Print only top 1")
parser.add_argument("-d", default=False, action="store_true", help="Save best")
parser.add_argument("--agents", nargs="*", default=[settings["defaults"]["agent"]])
parser.add_argument(
    "--dataset_loaders", nargs="*", default=[settings["defaults"]["dataset_loader"]]
)
parser.add_argument(
    "--metric_evaluator", default="CumulativeInteractionMetricEvaluator"
)
parser.add_argument("--metrics", nargs="*", default=[settings["defaults"]["metric"]])
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)
agents_search = yaml.load(open("./settings/agents_search.yaml"), Loader=yaml.SafeLoader)
dataset_agents_parameters = yaml.load(
    open("./settings/dataset_agents.yaml"), Loader=yaml.SafeLoader
)

settings["defaults"]["metric_evaluator"] = args.metric_evaluator
# print(args.b,args.m)

agents_search_parameters = yaml.load(open("./settings/agents_search.yaml"), Loader=yaml.SafeLoader)

utils.print_agent_search(args.agents,args.dataset_loaders, settings,dataset_agents_parameters, agents_search_parameters, args.metrics, args.d,args.t)
