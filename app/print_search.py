#!/usr/bin/python3

from os.path import dirname, realpath, sep, pardir
import os
import sys

sys.path.append(dirname(realpath(__file__)) + sep + pardir)

import json
import inquirer
import irec.value_functions
import irec.mf
from app import utils
import irec.evaluation_policies
import irec.metrics
import numpy as np
import scipy.sparse
import yaml

# from irec.metric_evaluators import CumulativeInteractionMetricsEvaluator
from irec.utils.dataset import Dataset

import matplotlib.pyplot as plt
from cycler import cycler
from collections import defaultdict
import argparse

settings = utils.load_settings()
parser = argparse.ArgumentParser()
parser.add_argument("-t", default=False, action="store_true", help="Print only top 1")
parser.add_argument("-d", default=False, action="store_true", help="Save best")
parser.add_argument("--agents", nargs="*", default=[settings["defaults"]["agent"]])
parser.add_argument(
    "--dataset_loaders", nargs="*", default=[settings["defaults"]["dataset_loader"]]
)
parser.add_argument(
    "--metric_evaluator", default=settings["defaults"]["metric_evaluator"]
)
parser.add_argument("--metrics", nargs="*", default=[settings["defaults"]["metric"]])
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)
agents_search = yaml.load(open("./settings/agents_search.yaml"), Loader=yaml.SafeLoader)

settings["defaults"]["metric_evaluator"] = args.metric_evaluator
# print(args.b,args.m)

metrics_classes = [irec.metrics.Hits]
metrics_names = ["Cumulative Hits"]


evaluation_policy_name = settings["defaults"]["evaluation_policy"]
evaluation_policy_parameters = settings["evaluation_policies"][evaluation_policy_name]
evaluation_policy = eval("irec.evaluation_policies." + evaluation_policy_name)(
    **evaluation_policy_parameters
)

interactors_classes_names_to_names = {
    k: v["name"] for k, v in settings["interactors_general_settings"].items()
}

# interactors_classes = ir.select_interactors()
# interactors_classes = [eval('irec.value_functions.'+interactor) for interactor in args.m]


metric_evaluator_parameters = settings["metric_evaluators"][
    settings["defaults"]["metric_evaluator"]
]

metric_class = eval("irec.metrics." + settings["defaults"]["metric"])

metric_evaluator = eval(
    "irec.metric_evaluators." + settings["defaults"]["metric_evaluator"]
)(None, **metric_evaluator_parameters)

datasets_metrics_values = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
)

for dataset_loader_name in args.dataset_loaders:
    settings["defaults"]["dataset_loader"] = dataset_loader_name
    traintest_dataset = utils.load_dataset_experiment(settings)

    for metric_name in args.metrics:
        for agent_name in args.agents:
            for agent_parameters in agents_search[agent_name]:
                settings["defaults"]["metric"] = metric_name
                settings["defaults"]["agent"] = agent_name
                settings["agents"][agent_name] = agent_parameters
                agent = utils.create_agent(agent_name, agent_parameters)
                agent_id = utils.get_agent_id(agent_name, agent_parameters)
                metric_values = utils.load_evaluation_experiment(settings)

                datasets_metrics_values[settings["defaults"]["dataset_loader"]][
                    settings["defaults"]["metric"]
                ][agent.name][json.dumps(agent_parameters)] = metric_values[-1]
# ','.join(map(lambda x: str(x[0])+'='+str(x[1]),list(parameters.items())))

# print(datasets_metrics_values)
for k1, v1 in datasets_metrics_values.items():
    for k2, v2 in v1.items():
        for k3, v3 in v2.items():
            values = np.array(list(v3.values()))
            keys = list(v3.keys())
            idxs = np.argsort(values)[::-1]
            keys = [keys[i] for i in idxs]
            values = [values[i] for i in idxs]
            if args.d:
                settings["agents_preprocessor_parameters"][k1][k3] = json.loads(keys[0])
            if args.t:
                print(f"{k3}:")
                # print('\tparameters:')
                agent_parameters, metric_value = json.loads(keys[0]), values[0]
                for name, value in agent_parameters.items():
                    print(f"\t\t{name}: {value}")
            else:
                for k4, v4 in zip(keys, values):
                    k4 = yaml.safe_load(k4)
                    # k4 = ','.join(map(lambda x: str(x[0])+'='+str(x[1]),list(k4.items())))
                    print(f"{k3}({k4}) {v4:.5f}")

if args.d:
    print("Saved parameters!")
    open("settings" + sep + "agents_preprocessor_parameters.yaml", "w").write(
        yaml.dump(utils.default_to_regular(settings["agents_preprocessor_parameters"]))
    )
