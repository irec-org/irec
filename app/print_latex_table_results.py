#!/usr/bin/python3
from os.path import dirname, realpath
import yaml
import argparse

from irec.app import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--evaluation_policy")
parser.add_argument("--dataset_loaders", nargs="*")
parser.add_argument("--agents", nargs="*")
parser.add_argument("--metrics", nargs="*")
parser.add_argument("--metric_evaluator")
parser.add_argument("-r", type=str, default=None)
parser.add_argument("--type", default=None)
parser.add_argument("--dump", default=False, action="store_true")
settings = utils.load_settings(dirname(realpath(__file__)))
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

dataset_agents_parameters = yaml.load(
    open("./settings/dataset_agents.yaml"), Loader=yaml.SafeLoader
)

utils.print_results_latex_table(args.agents,args.dataset_loaders,settings,dataset_agents_parameters, args.metrics, args.r,args.dump,args.type)
