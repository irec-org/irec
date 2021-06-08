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
# parser.add_argument('-f', default='')
# parser.add_argument('-f', default=False, action='store_true')
args = parser.parse_args()
import inquirer
import value_functions
from lib.utils.InteractorRunner import InteractorRunner
import joblib
import concurrent.futures
from lib.utils.DatasetManager import DatasetManager
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import mf
import lib.utils.utils as util
# from util import DatasetFormatter, MetricsEvaluator
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
# import recommenders
import evaluation_policies
import yaml
import lib.utils.dataset
from lib.utils.InteractorCache import InteractorCache
from lib.utils.PersistentDataManager import PersistentDataManager
import metrics

metrics_classes = [metrics.Hits]

interactors_preprocessor_parameters = yaml.load(
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
                      interactors_preprocessor_parameters,
                      evaluation_policies_parameters)
interactors_classes = [
    eval('value_functions.' + interactor) for interactor in args.m
]
# history_rates_to_train = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

history_rates_to_train = [0.1]
# history_rates_to_train = [0.1,0.3,0.5,0.6]

# history_rates_to_train = [0.1,0.3,0.5,0.6,0.8]
# history_rates_to_train = [0.8]
# def process(history_rate, dataset_preprocessor, dataset, consumption_matrix,
            # dm):




# with concurrent.futures.ProcessPoolExecutor(
        # max_workers=args.num_tasks) as executor:
    # futures = set()
for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)
    dm.load()
    data = np.vstack(
        (dm.dataset_preprocessed[0].data, dm.dataset_preprocessed[1].data))
    dataset = utils.dataset.Dataset(data)
    dataset.update_from_data()
    dataset.update_num_total_users_items()
    consumption_matrix = scipy.sparse.csr_matrix(
        (dataset.data[:, 2],
         (dataset.data[:, 0].astype(int), dataset.data[:, 1].astype(int))),
        shape=(dataset.num_total_users, dataset.num_total_items))
    for history_rate in history_rates_to_train:
        print('%.2f%% of history' % (history_rate * 100))
        for interactor_class in interactors_classes:
            metric_evaluator = metrics.TotalMetricsEvaluator(dataset, metrics_classes)
            itr = interactor_class(**interactors_preprocessor_parameters[
                dataset_preprocessor['name']][interactor_class.__name__]['parameters'])

            start_evaluation_policy = eval('evaluation_policy.' + args.estart)(
                **evaluation_policies_parameters[args.estart])
            start_evaluation_policy.recommend_test_data_rate_limit = history_rate
            # no need history rate s but i will put it because of consistency
            file_name = 's_' + str(history_rate) + '_' + InteractorCache().get_id(
                dm, start_evaluation_policy, itr)

            pdm = PersistentDataManager(directory='results',)
            if pdm.file_exists(file_name):
                print("File already exists")
                history_items_recommended = pdm.load(file_name)
                history_items_recommended = np.array(history_items_recommended)
                num_users_test = len(np.unique(history_items_recommended[:,0]))
                num_interactions = len(history_items_recommended)/num_users_test
                file_name = 's_num_interactions_' + str(history_rate) + '_' + InteractorCache().get_id(
                    dm, start_evaluation_policy, itr)
                pdm_out = PersistentDataManager(directory='metrics',extension_name='.txt')
                fp = pdm_out.get_fp(file_name)
                # print(fp)
                # raise SystemError
                util.create_path_to_file(fp)
                with open(fp,'w+') as fout:
                    fout.write(str(num_interactions))

                print(fp)
            else:
                print(f"File doesnt exists {file_name}")
                raise SystemError

            itr = interactor_class(**interactors_preprocessor_parameters[
                dataset_preprocessor['name']][interactor_class.__name__]['parameters'])

            last_evaluation_policy = eval('evaluation_policy.' + args.elast)(
                **evaluation_policies_parameters[args.elast])
            file_name = 'e_' + str(history_rate) + '_' + InteractorCache().get_id(
                dm, last_evaluation_policy, itr)

            if pdm.file_exists(file_name):
                print("File already exists")
                print(pdm.get_fp(file_name))
                history_items_recommended = pdm.load(file_name)
                metrics_values = metric_evaluator.evaluate(history_items_recommended)
                metrics_pdm = PersistentDataManager(directory='metrics')
                for metric_name, metric_values in metrics_values.items():
                    metrics_pdm.save(
                        os.path.join(InteractorCache().get_id(dm, last_evaluation_policy, itr),
                                     metric_evaluator.get_id(), metric_name+'_'+str(history_rate)), metric_values)
                pass
            else:
                print(f"File doenst exists {file_name}")
                raise SystemError
