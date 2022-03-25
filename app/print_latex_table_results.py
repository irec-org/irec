#!/usr/bin/python3
import argparse
import yaml
import sys
from os.path import dirname, realpath, sep, pardir
# sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "irec")
# import irec
# print(sys.path)
from irec.app import utils

settings = utils.load_settings(dirname(realpath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--evaluation_policy", default=settings["defaults"]["evaluation_policy"]
)
parser.add_argument("--dataset_loaders", nargs="*")
parser.add_argument("--agents", nargs="*")
parser.add_argument("--metrics", nargs="*")
parser.add_argument(
    "--metric_evaluator", default=settings["defaults"]["metric_evaluator"]
)
parser.add_argument("-r", type=str, default=None)
parser.add_argument("--type", default=None)
parser.add_argument("--dump", default=False, action="store_true")
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

dataset_agents_parameters = yaml.load(
    open("./settings/dataset_agents.yaml"), Loader=yaml.SafeLoader
)

settings["defaults"]["metric_evaluator"] = args.metric_evaluator
settings["defaults"]["evaluation_policy"] = args.evaluation_policy

utils.print_results_latex_table(
    args.agents,
    args.dataset_loaders,
    settings,
    dataset_agents_parameters,
    args.metrics,
    args.r,
    args.dump,
    args.type,
)
