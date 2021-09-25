#!/bin/python3

import os
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
from app.constants import WORKINGDIR
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from app import utils

# from app import utils
import subprocess
import argparse

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

args = parser.parse_args()
evaluation_policy_name = args.evaluation_policy

with ProcessPoolExecutor(max_workers=args.tasks) as executor:
    futures = set()
    for agent_name in args.agents:
        for dataset_loader_name in args.dataset_loaders:

            f = executor.submit(
                subprocess.run,
                "./run_agent_best.py --dataset_loader '{}' --agent '{}' --evaluation_policy '{}'".format(
                    dataset_loader_name,
                    agent_name,
                    evaluation_policy_name,
                ),
                cwd=WORKINGDIR,
                shell=True,
            )
            futures.add(f)
            if len(futures) >= args.tasks:
                completed, futures = wait(futures, return_when=FIRST_COMPLETED)
    for f in futures:
        f.result()
