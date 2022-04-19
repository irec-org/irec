#!/usr/bin/python3

from os.path import dirname, realpath
import yaml
import os
import argparse

from irec.connector import utils
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
parser.add_argument("--metrics", nargs="*", default=[settings["defaults"]["metric"]])
parser.add_argument(
    "--metric_evaluator", default="CumulativeInteractionMetricEvaluator"
)
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

agents_search = yaml.load(open("./settings/agents_search.yaml"), Loader=yaml.SafeLoader)

settings["defaults"]["evaluation_policy"] = args.evaluation_policy
settings["defaults"]["metric_evaluator"] = args.metric_evaluator

utils.eval_agent_search(args.agents,args.dataset_loaders,
        settings,agents_search,args.metrics, args.tasks)
