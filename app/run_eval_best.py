#!/usr/bin/python3

import os
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from app import utils
import numpy as np
from irec.utils.dataset import Dataset

# from app import utils
import yaml
import argparse
import copy

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


dataset_agents = yaml.load(
    open("./settings/dataset_agents.yaml"), Loader=yaml.SafeLoader
)

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
            settings["defaults"]["evaluation_policy"] = args.evaluation_policy
            settings["defaults"]["agent"] = agent_name
            settings["agents"][agent_name] = dataset_agents[dataset_loader_name][
                agent_name
            ]

            for metric_name in args.metrics:
                settings["defaults"]["metric"] = metric_name
                settings["defaults"]["metric_evaluator"] = args.metric_evaluator
                f = executor.submit(
                    utils.evaluate_itr, dataset, copy.deepcopy(settings)
                )
                futures.add(f)
                if len(futures) >= args.tasks:
                    completed, futures = wait(futures, return_when=FIRST_COMPLETED)
    for f in futures:
        f.result()
