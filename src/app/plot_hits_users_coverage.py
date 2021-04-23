from os.path import dirname, realpath, sep, pardir
import pickle
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
import matplotlib.ticker as mtick


parser = argparse.ArgumentParser()
parser.add_argument('-r', type=str, default=None)
# parser.add_argument('-i', default=[5,10,20,50,100],nargs='*')
parser.add_argument('-m',nargs='*')
parser.add_argument('-b',nargs='*')
parser.add_argument('--dump', default=False, action='store_true')
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
                metric_class_name][interactors_general_settings[itr_class.__name__]['name']]= np.mean(metric_values[-1])
            datasets_metrics_users_values[dataset_preprocessor['name']][
                metric_class_name][interactors_general_settings[itr_class.__name__]['name']]= np.array(metric_values[-1])


def plot_hits_users_coverage(dataset_methods_users_hits, title="Users Coverage x Cum. Precision", xlabel="Cumulative Precision", ylabel='Users Coverage - P(X ≤ x)'):
    
    fig, axs = plt.subplots(1,3)
    fig.set_size_inches(13, 4)
    fig.subplots_adjust(wspace=0.3)
    # fig.suptitle(title, fontsize=14, y=0.88)

    plt.rcParams.update({'font.size': 10})
    plt.subplots_adjust(top=0.80)

    for index in range(0, len(axs)):
        axs[index].set_xlabel(xlabel, fontsize='medium')
        axs[index].set_ylabel(ylabel, fontsize='medium')
        axs[index].yaxis.set_major_formatter(mtick.PercentFormatter())

    colors = ["#FFFF66", "#66FF66", "#66FFFF", "#6600CC", "#FF66B2", "#808080", "#666600", "#FF0000", "tab:pink"]
    markers = ["P", "1", ",", ">", "p", "3", "2", "*", "."]

    position_sub_graphs = [.21, .49, .77]
    zoomx = [(0, 80), (0, 60), (0, 30)]
    zoomy = [(0, 20), (0, 20), (0, 20)]
    stepx = [20, 10, 5]
    for index, (dataset, methods_users_hits) in enumerate(dataset_methods_users_hits.items()):
        for i, (method, hits) in enumerate(methods_users_hits.items()):
            
            method = method.replace(" (PMF)", "")
            
            axs[index].set_title(dataset)
            axs[index].plot(list(range(0, len(hits))), hits, colors[i], label=method, marker=markers[i], markevery=[i for i in range(0, 100, 10)])
    
            sub_axes = plt.axes([position_sub_graphs[index], .52, .12, .25]) 
            sub_axes.plot(list(range(0, len(hits))), hits, colors[i], marker=markers[i], markevery=[i for i in range(0, 100, 10)])
            sub_axes.set_xlim(zoomx[index])
            sub_axes.set_ylim(zoomy[index])
            sub_axes.set_xticks(list(range(zoomx[index][0], zoomx[index][1], stepx[index])))
            sub_axes.set_yticks(list(range(0, 25, 5)))
            sub_axes.tick_params(labelsize=8)
            sub_axes.set_yticklabels([str(x)+"%" for x in [0, 5, 10, 15, 20]])

    axs[1].legend(bbox_to_anchor=(1.95, 1.25), ncol=i+1) 
    axs[2].set_yticks(list(range(0, len(hits)+20, 20)))
    
    return fig
 

def count_hits(methods_users_hits, num_items=100):
    list_hits = dict.fromkeys(methods_users_hits, [])  
    num_users = len(methods_users_hits["WSCB"])

    for method, hits in methods_users_hits.items():
        list_hits[method] = [(list(hits).count(i)/num_users)*100 for i in range(0, num_items)]

    return list_hits


dataset_methods_users_hits = {}
for dataset_preprocessor in datasets_preprocessors:
    methods_users_hits = dict(datasets_metrics_users_values[dataset_preprocessor['name']]['Hits'])
    dataset_methods_users_hits[dataset_preprocessor["name"]] = {k:(np.cumsum(np.array(v)[::-1]))[::-1]-np.array(v) for k,v in count_hits(methods_users_hits).items()}

fig = plot_hits_users_coverage(dataset_methods_users_hits, f"Users Coverage $\\times$ Hits ({dataset_preprocessor['name']})",ylabel='Users Coverage - P(X > x)')

fig.savefig(os.path.join(DirectoryDependent().DIRS["img"],f'plot_hits_users_coverage_ccdf_{dataset_preprocessor["name"]}.png'),bbox_inches = 'tight')

# plot_hits_users_coverage(list_hits)
# for dataset_preprocessor in datasets_preprocessors:
#     methods_users_hits = dict(datasets_metrics_users_values[dataset_preprocessor['name']]['Hits'])
#     if args.dump:
#         with open('methods_users_hits_{}.pickle'.format(dataset_preprocessor['name']),'wb') as f:
#             pickle.dump(methods_users_hits,f)
#         # f.write(str(methods_users_hits))
#     fig = plot_hits_users_coverage(count_hits(methods_users_hits),f"Users Coverage $\\times$ Hits ({dataset_preprocessor['name']})",ylabel="Users Coverage %")

#     fig.savefig(os.path.join(DirectoryDependent().DIRS["img"],f'plot_hits_users_coverage_{dataset_preprocessor["name"]}.png'),bbox_inches = 'tight')
# # 
#     fig = plot_hits_users_coverage({k:np.cumsum(v) for k,v in count_hits(methods_users_hits).items()},f"Users Coverage $\\times$ Hits ({dataset_preprocessor['name']})",ylabel='Users Coverage - P(X ≤ x)')

#     fig.savefig(os.path.join(DirectoryDependent().DIRS["img"],f'plot_hits_users_coverage_cdf_{dataset_preprocessor["name"]}.png'),bbox_inches = 'tight')

#     fig = plot_hits_users_coverage({k:(np.cumsum(np.array(v)[::-1]))[::-1]-np.array(v) for k,v in count_hits(methods_users_hits).items()},f"Users Coverage $\\times$ Hits ({dataset_preprocessor['name']})",ylabel='Users Coverage - P(X > x)')

#     fig.savefig(os.path.join(DirectoryDependent().DIRS["img"],f'plot_hits_users_coverage_ccdf_{dataset_preprocessor["name"]}.png'),bbox_inches = 'tight')
    

# datasets_metrics_users_values[]
# datasets_metrics_gain = defaultdict(
        # lambda: defaultdict(lambda: defaultdict(lambda: ['']*len(nums_interactions_to_show))))

# datasets_metrics_best = defaultdict(
        # lambda: defaultdict(lambda: defaultdict(lambda:[False]*len(nums_interactions_to_show))))
