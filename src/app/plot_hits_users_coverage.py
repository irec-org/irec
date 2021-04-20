from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
import copy
import scipy
import interactors
import mf
from utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from utils.DatasetManager import DatasetManager
import yaml
from metric import CumulativeInteractionMetricsEvaluator, UserCumulativeInteractionMetricsEvaluator
from utils.dataset import Dataset
from utils.PersistentDataManager import PersistentDataManager
from utils.InteractorCache import InteractorCache
import metric
import matplotlib.pyplot as plt
from utils.DirectoryDependent import DirectoryDependent
from cycler import cycler
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-r', type=str, default=None)
# parser.add_argument('-i', default=[5,10,20,50,100],nargs='*')
parser.add_argument('-m',nargs='*')
parser.add_argument('-b',nargs='*')
args = parser.parse_args()

plt.rcParams['axes.prop_cycle'] = cycler(color='krbgmyc')
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 15

# metrics_classes = [metric.Hits, metric.Recall]
metrics_classes = [metric.Hits]
        # metric.Recall ,
        # metric.EPC,
        # metric.UsersCoverage, 
        # metric.ILD,
        # metric.GiniCoefficientInv,
        # ]
# metrics_names = ['Cumulative Precision', 
        # 'Cumulative Recall', 
        # 'Cumulative EPC', 
        # 'Cumulative Users Coverage',
        # 'Cumulative ILD',
        # '1-(Gini-Index)'
        # ]

# dm = DatasetManager()
# datasets_preprocessors = dm.request_datasets_preprocessors()
# print(datasets_preprocessors_classes)

with open("settings"+sep+"datasets_preprocessors_parameters.yaml") as f:
    loader = yaml.SafeLoader
    datasets_preprocessors = yaml.load(f,Loader=loader)

    datasets_preprocessors = {setting['name']: setting
                              for setting in datasets_preprocessors}
evaluation_policies_parameters = yaml.load(
    open("settings" + sep + "evaluation_policies_parameters.yaml"),
    Loader=yaml.SafeLoader)
interactors_preprocessor_paramaters = yaml.load(
    open("settings" + sep + "interactors_preprocessor_parameters.yaml"),
    Loader=yaml.SafeLoader)
interactors_general_settings = yaml.load(
    open("settings" + sep + "interactors_general_settings.yaml"),
    Loader=yaml.SafeLoader)

evaluation_policies_parameters = yaml.load(
    open("settings" + sep + "evaluation_policies_parameters.yaml"),
    Loader=yaml.SafeLoader)

interactors_classes_names_to_names = {
    k: v['name'] for k, v in interactors_general_settings.items()
}

dm = DatasetManager()
datasets_preprocessors = [datasets_preprocessors[base] for base in args.b]
ir = InteractorRunner(dm, interactors_general_settings,
                      interactors_preprocessor_paramaters,
                      evaluation_policies_parameters)
interactors_classes = [eval('interactors.'+interactor) for interactor in args.m]

# ir = InteractorRunner(dm, interactors_general_settings,
                      # interactors_preprocessor_paramaters,
                      # evaluation_policies_parameters)
# interactors_classes = ir.select_interactors()

metrics_evaluator = UserCumulativeInteractionMetricsEvaluator(None, metrics_classes)

evaluation_policy = ir.get_interactors_evaluation_policy()

# nums_interactions_to_show = list(map(int,args.i))




datasets_metrics_values = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list)))

datasets_metrics_users_values = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list)))

for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)

    for metric_class_name in map(lambda x: x.__name__, metrics_classes):
        for itr_class in interactors_classes:
            itr = ir.create_interactor(itr_class)
            pdm = PersistentDataManager(directory='results')

            metrics_pdm = PersistentDataManager(directory='metrics')
            metric_values = metrics_pdm.load(
                os.path.join(
                    InteractorCache().get_id(dm, evaluation_policy, itr),
                    metrics_evaluator.get_id(), metric_class_name))
            # print(len(metric_values))
            datasets_metrics_values[dataset_preprocessor['name']][
                metric_class_name][interactors_general_settings[itr_class.__name__]['name']]=                  np.mean(metric_values[-1])
            datasets_metrics_users_values[dataset_preprocessor['name']][
                metric_class_name][interactors_general_settings[itr_class.__name__]['name']]=                   np.array(metric_values[-1])


def plot_hits_users_coverage(methods_users_hits, title="Users Coverage x Hits"):
    """
    Args:
      methods_users_hits (dict): Dict where keys are names of methods 
      and values are np.array containing number of hits achieved by each user
    Returns:
      matplotlib.pyplot.figure: A figure of the users-coverage
    """
    
    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 6)
    fig.suptitle(title, fontsize=20, y=0.88)

    plt.rcParams.update({'font.size': 20})
    plt.subplots_adjust(top=0.80)

    ax1.set_xlabel("Hits", fontsize='medium')
    ax1.set_ylabel("Users Coverage %", fontsize='medium')
    
    colors = ["b--", "g--", "y--", "b--", "c--", "m--", "r-*", "b-d", "g-o"]
    for i, (method, hits) in enumerate(methods_users_hits.items()):
        ax1.plot(list(range(0, len(hits))), hits, colors[i], label=method)

    ax1.legend(ncol=1)
    ax1.tick_params(labelsize=18)
    plt.xticks(list(range(0, len(hits)+10, 10)))
    plt.yticks(list(range(0, len(hits)+10, 10)))

    return fig


def count_hits(methods_users_hits, num_items=100):
    list_hits = dict.fromkeys(methods_users_hits, [])
    num_users = len(methods_users_hits["WSCB"])

    for method, hits in methods_users_hits.items():
        list_hits[method] = [(list(hits).count(i)/num_users)*100 for i in range(0, num_items)]

    return list_hits

# plot_hits_users_coverage(list_hits)
for dataset_preprocessor in datasets_preprocessors:
    methods_users_hits = dict(datasets_metrics_users_values[dataset_preprocessor['name']]['Hits'])
    # with open('outlk.txt','w') as f:
        # f.write(str(methods_users_hits))
    fig = plot_hits_users_coverage(count_hits(methods_users_hits),f"Users Coverage $\\times$ Hits ({dataset_preprocessor['name']})")

    fig.savefig(os.path.join(DirectoryDependent().DIRS["img"],f'plot_hits_users_coverage_{dataset_preprocessor["name"]}.png'),bbox_inches = 'tight')
    

# datasets_metrics_users_values[]
# datasets_metrics_gain = defaultdict(
        # lambda: defaultdict(lambda: defaultdict(lambda: ['']*len(nums_interactions_to_show))))

# datasets_metrics_best = defaultdict(
        # lambda: defaultdict(lambda: defaultdict(lambda:[False]*len(nums_interactions_to_show))))
