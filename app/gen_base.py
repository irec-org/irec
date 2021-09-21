from os.path import dirname, realpath, sep, pardir
import tempfile
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir)

from app import constants
from app import utils
import argparse
from irec.utils.DatasetLoaderFactory import DatasetLoaderFactory
from mlflow import log_param, log_artifact
import mlflow
import pickle

dataset_loader_factory= DatasetLoaderFactory()
parser = argparse.ArgumentParser()
parser.add_argument('-b')
args = parser.parse_args()
settings = utils.load_settings()

dataset_loader_settings =settings['dataset_loaders'][args.b]

mlflow.set_experiment("dataset")

with mlflow.start_run() as run:
    utils.log_custom_parameters(utils.parameters_normalize(constants.DATASET_PARAMETERS_PREFIX,args.b,dataset_loader_settings))
    # client.log_param()
    # for k,v in dataset_loader_settings.items():
        # log_param(k,v)

    dataset_loader = dataset_loader_factory.create(args.b,dataset_loader_settings)
    dataset = dataset_loader.load()

    fname = './tmp/dataset.pickle'
    utils.log_custom_artifact(fname,dataset)
