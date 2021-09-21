import os
from os.path import dirname, realpath, sep, pardir
import sys
from copy import copy
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "irec")
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
import matplotlib
from irec.utils.DirectoryDependent import DirectoryDependent
from collections import defaultdict
import value_functions
import matplotlib.pyplot as plt
from irec.utils.InteractorRunner import InteractorRunner
import joblib
import concurrent.futures
from irec.utils.DatasetManager import DatasetManager
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import mf
import irec.utils.utils as util
# from util import DatasetFormatter, MetricsEvaluator
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
# import recommenders
import evaluation_policies
import yaml
import irec.utils.dataset
from irec.utils.InteractorCache import InteractorCache
from irec.utils.PersistentDataManager import PersistentDataManager
import metrics

datasets_metrics_rate_values = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
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
        setting['name']: setting
        for setting in datasets_preprocessors
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
# history_rates_to_train = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# history_rates_to_train = [0.1,0.3,0.5,0.6]

# history_rates_to_train = [0.1,0.3,0.5,0.6,0.8]
history_rates_to_train = [0.1, 0.3, 0.5, 0.6]

for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)
    for history_rate in history_rates_to_train:
        print('%.2f%% of history' % (history_rate * 100))
        for interactor_class in interactors_classes:
            metric_evaluator = metrics.TotalMetricsEvaluator(
                None, metrics_classes)
            itr = interactor_class(**interactors_preprocessor_parameters[
                dataset_preprocessor['name']][interactor_class.__name__]
                                   ['parameters'])

            start_evaluation_policy = eval('evaluation_policy.' + args.estart)(
                **evaluation_policies_parameters[args.estart])
            start_evaluation_policy.recommend_test_data_rate_limit = history_rate
            file_name = 's_num_interactions_' + str(
                history_rate) + '_' + InteractorCache().get_id(
                    dm, start_evaluation_policy, itr)

            # file_name = 's_num_interactions_' + str(history_rate) + '_' + InteractorCache().get_id(
            # dm, start_evaluation_policy, itr)
            # pdm_out = PersistentDataManager(directory='metrics',extension_name='.txt')
            # fp = pdm_out.get_fp(file_name)
            # print(fp)
            # raise SystemError
            # util.create_path_to_file(fp)
            # print(history_rate)

            pdm_out = PersistentDataManager(directory='metrics',
                                            extension_name='.txt')
            fp = pdm_out.get_fp(file_name)
            with open(fp, 'r') as fi:
                datasets_metrics_rate_values[
                    dataset_preprocessor['name']]['num_interactions'][
                        interactor_class.__name__][history_rate] = float(
                            fi.read())
            # num_interactions = float(open(fp, 'r').read())

            itr = interactor_class(**interactors_preprocessor_parameters[
                dataset_preprocessor['name']][interactor_class.__name__]
                                   ['parameters'])

            last_evaluation_policy = eval('evaluation_policy.' + args.elast)(
                **evaluation_policies_parameters[args.elast])

            metrics_pdm = PersistentDataManager(directory='metrics')
            metrics_values = dict()
            for metric_name in list(map(lambda x: x.__name__,
                                        metrics_classes)):
                metrics_values[metric_name] = metrics_pdm.load(
                    os.path.join(
                        InteractorCache().get_id(dm,
                                                 last_evaluation_policy, itr),
                        metric_evaluator.get_id(),
                        metric_name + '_' + str(history_rate)))
                # datasets_metrics_values
                metrics_values[metric_name] = np.mean(
                    metrics_values[metric_name][0])
                datasets_metrics_rate_values[
                    dataset_preprocessor['name']][metric_name][
                        interactor_class.
                        __name__][history_rate] = metrics_values[metric_name]

print(datasets_metrics_rate_values)
font = {
    'family': 'normal',
    # 'weight' : 'bold',
    'size': 20
}

matplotlib.rc('font', **font)
fig, axs = plt.subplots(nrows=1 + len(metrics_classes),
                        ncols=len(args.b),
                        figsize=(12, 12))

j = 0
# MAUT = 0
for dataset, metrics_rate_values in datasets_metrics_rate_values.items():
    i = 0
    for metrics, itr_rate_values in metrics_rate_values.items():
        if metric == 'Hits':
            metric = 'Cum. Precision'
        if metric == 'num_interactions':
            metric = 'Number of interactions'
        else:
            metric += '@10'
        # metric = metric == 'num_interactions' ? 'Number of interactions' : metric
        axs[i].set_ylabel(metric)
        axs[i].set_xlabel('Recall')
        axs[i].set_title(dataset)
        print(itr_rate_values)
        for itr_name, rate_values in itr_rate_values.items():
            x = list(rate_values.keys())
            y = list(rate_values.values())
            print(x)
            print(y)
            axs[i].plot(x, y, label=itr_name)

        i += 1
    # axs
    j += 1

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles,
           labels,
           loc='upper center',
           ncol=5,
           bbox_to_anchor=(0.75, 1.05),
           fancybox=True,
           shadow=True)
fig.tight_layout()
fig.savefig(os.path.join(DirectoryDependent().DIRS["img"],
                         f'plot_custom_train.png'),
            bbox_inches='tight')
