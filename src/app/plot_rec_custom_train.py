import os
from os.path import dirname, realpath, sep, pardir
import sys
from copy import copy
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--forced_run', default=False, action='store_true')
# parser.add_argument('--parallel', default=False, action='store_true')
parser.add_argument('-m', nargs='*')
parser.add_argument('-b', nargs='*')
parser.add_argument('--num_tasks', type=int, default=os.cpu_count())
parser.add_argument('-estart', default='LimitedInteraction')
parser.add_argument('-elast', default='Interaction')
args = parser.parse_args()
import inquirer
import interactors
from utils.InteractorRunner import InteractorRunner
import joblib
import concurrent.futures
from utils.DatasetManager import DatasetManager
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import mf
import utils.util as util
# from util import DatasetFormatter, MetricsEvaluator
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
# import recommenders
import evaluation_policy
import yaml
import utils.dataset
from utils.InteractorCache import InteractorCache
from utils.PersistentDataManager import PersistentDataManager
import metric

metrics_classes = [metric.Hits]

interactors_preprocessor_paramaters = yaml.load(
    open("settings" + sep + "interactors_preprocessor_parameters.yaml"),
    Loader=yaml.SafeLoader)
interactors_general_settings = yaml.load(
    open("settings" + sep + "interactors_general_settings.yaml"),
    Loader=yaml.SafeLoader)

evaluation_policies_parameters = yaml.load(
    open("settings" + sep + "evaluation_policies_parameters.yaml"),
    Loader=yaml.SafeLoader)

with open("settings" + sep + "datasets_preprocessors_parameters.yaml") as f:
    loader = yaml.SafeLoader
    datasets_preprocessors = yaml.load(f, Loader=loader)

    datasets_preprocessors = {
        setting['name']: setting for setting in datasets_preprocessors
    }

dm = DatasetManager()
datasets_preprocessors = [datasets_preprocessors[base] for base in args.b]
ir = InteractorRunner(None, interactors_general_settings,
                      interactors_preprocessor_paramaters,
                      evaluation_policies_parameters)
interactors_classes = [
    eval('interactors.' + interactor) for interactor in args.m
]
history_rates_to_train = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)
    for history_rate in history_rates_to_train:
        print('%.2f%% of history' % (history_rate * 100))
        for interactor_class in interactors_classes:
            metric_evaluator = metric.TotalMetricsEvaluator(
                None, metrics_classes)
            itr = interactor_class(**interactors_preprocessor_paramaters[
                dataset_preprocessor['name']][interactor_class.__name__]
                                   ['parameters'])

            start_evaluation_policy = eval('evaluation_policy.' + args.estart)(
                **evaluation_policies_parameters[args.estart])
            start_evaluation_policy.recommend_test_data_rate_limit = history_rate
            file_name = 's_num_interactions_' + str(
                history_rate) + '_' + InteractorCache().get_id(
                    dm, start_evaluation_policy, itr)
            pdm_out = PersistentDataManager(directory='metrics',
                                            extension_name='.txt')
            fp = pdm_out.get_fp(file_name)
            num_interactions = float(open(fp, 'r').read())

            itr = interactor_class(**interactors_preprocessor_paramaters[
                dataset_preprocessor['name']][interactor_class.__name__]
                                   ['parameters'])

            last_evaluation_policy = eval('evaluation_policy.' + args.elast)(
                **evaluation_policies_parameters[args.elast])

            metrics_pdm = PersistentDataManager(directory='metrics')
            metrics_values = dict()
            for metric_name in list(map(lambda x: x.__name__, metrics_classes)):
                metrics_values[metric_name] = metrics_pdm.load(
                    os.path.join(
                        InteractorCache().get_id(dm, last_evaluation_policy, itr),
                        metric_evaluator.get_id(), metric_name))
