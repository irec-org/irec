#!/usr/bin/python3
from os.path import dirname, realpath, sep, pardir
import sys

sys.path.append(dirname(realpath(__file__)) + sep + pardir)

from app import constants
from app import utils
import argparse
from irec.utils.DatasetLoaderFactory import DatasetLoaderFactory
import mlflow

settings = utils.load_settings()
dataset_loader_factory = DatasetLoaderFactory()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_loaders", nargs="*", default=[settings["defaults"]["dataset_loader"]]
)
args = parser.parse_args()

utils.set_experiments(settings)

for dataset_name in args.dataset_loaders:
    dataset_loader_settings = settings["dataset_loaders"][dataset_name]

    mlflow.set_experiment("dataset")

    with mlflow.start_run() as run:
        utils.log_custom_parameters(
            utils.parameters_normalize(
                constants.DATASET_PARAMETERS_PREFIX,
                dataset_name,
                dataset_loader_settings,
            )
        )
        # client.log_param()
        # for k,v in dataset_loader_settings.items():
        # log_param(k,v)

        dataset_loader = dataset_loader_factory.create(
            dataset_name, dataset_loader_settings
        )
        dataset = dataset_loader.load()

        fname = "./tmp/dataset.pickle"
        utils.log_custom_artifact(fname, dataset)
