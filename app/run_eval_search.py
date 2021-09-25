#!/bin/python3

from os.path import dirname, realpath
import os
import sys

# sys.path.append(dirname(dirname(realpath(__file__))))
sys.path.append(dirname(dirname(realpath(__file__))))
# print(os.path.join(dirname(realpath(__file__)), "..", "app"))
# from utils import flatten_dict
from app import utils

import numpy as np
from irec.utils.dataset import Dataset

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
parser.add_argument("--metrics", nargs="*", default=[settings["defaults"]["metric"]])
parser.add_argument(
    "--metric_evaluator", default=settings["defaults"]["metric_evaluator"]
)
args = parser.parse_args()

agents_search = yaml.load(open("./settings/agents_search.yaml"), Loader=yaml.SafeLoader)

settings["defaults"]["evaluation_policy"] = args.evaluation_policy
settings["defaults"]["metric_evaluator"] = args.metric_evaluator
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
        traintest_dataset = utils.load_dataset_experiment(settings)

        data = np.vstack((traintest_dataset.train.data, traintest_dataset.test.data))

        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        for agent_name in args.agents:
            settings["defaults"]["agent"] = agent_name
            for metric_name in args.metrics:
                for agent_og_parameters in agents_search[agent_name]:
                    settings["agents"][agent_name] = agent_og_parameters
                    settings["defaults"]["metric"] = metric_name
                    # agent_og_parameters = dataset_agents[dataset_loader_name][agent_name]
                    f = executor.submit(
                        utils.evaluate_itr, dataset, copy.deepcopy(settings)
                    )
                    futures.add(f)
                    if len(futures) >= args.tasks:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
    for f in futures:
        f.result()
