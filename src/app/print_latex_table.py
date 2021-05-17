from os.path import dirname, realpath, sep, pardir
import pickle
import os
import sys
import json
import utils
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
import copy
import scipy
import interactors
import mf
from lib.utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from lib.utils.DatasetManager import DatasetManager
import yaml
import lib.evaluation_policies
from metrics import CumulativeInteractionMetricsEvaluator, UserCumulativeInteractionMetricsEvaluator
from lib.utils.dataset import Dataset
from lib.utils.PersistentDataManager import PersistentDataManager
from lib.utils.InteractorCache import InteractorCache
import metrics
import matplotlib.pyplot as plt
from lib.utils.DirectoryDependent import DirectoryDependent
from cycler import cycler
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-r', type=str, default=None)
parser.add_argument('-i', default=[5,10,20,50,100],nargs='*')
parser.add_argument('-m',nargs='*')
parser.add_argument('-b',nargs='*')
parser.add_argument('--type',default=None)
parser.add_argument('--dump', default=False, action='store_true')
settings = utils.load_settings()
utils.load_settings_to_parser(settings,parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings,args)

plt.rcParams['axes.prop_cycle'] = cycler(color='krbgmyc')
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 15

# metrics_classes = [metrics.Hits, metrics.Recall]
metrics_classes = [metrics.Hits,
        metrics.Recall ,
        metrics.EPC,
        metrics.UsersCoverage, 
        metrics.ILD,
        metrics.GiniCoefficientInv,
        ]
metrics_classes_names = list(map(lambda x: x.__name__, metrics_classes))
metrics_names = ['Cumulative Precision', 
        'Cumulative Recall', 
        'Cumulative EPC', 
        'Cumulative Users Coverage',
        'Cumulative ILD',
        '1-(Gini-Index)'
        ]
metrics_weights = {'Hits': 0.3,'Recall':0.3,'EPC':0.1,'UsersCoverage':0.1,'ILD':0.1,'GiniCoefficientInv':0.1}
# metrics_weights = {'Hits': 0.3,'Recall':0.3,'EPC':0.16666,'UsersCoverage':0.16666,'ILD':0.16666}
# metrics_weights = {'Hits': 0.25,'Recall':0.25,'EPC':0.125,'UsersCoverage':0.125,'ILD':0.125,'GiniCoefficientInv':0.125}
# metrics_weights ={i: 1/len(metrics_classes_names) for i in metrics_classes_names}


interactors_classes_names_to_names = {
    k: v['name'] for k, v in settings['interactors_general_settings'].items()
}

dm = DatasetManager()
datasets_preprocessors = [settings['datasets_preprocessors_parameters'][base] for base in args.b]
ir = InteractorRunner(dm, settings['interactors_general_settings'],
                      settings['interactors_preprocessor_parameters'],
                      settings['evaluation_policies_parameters'])
interactors_classes = [eval('lib.interactors.'+interactor) for interactor in args.m]

# ir = InteractorRunner(dm, interactors_general_settings,
                      # interactors_preprocessor_parameters,
                      # evaluation_policies_parameters)
# interactors_classes = ir.select_interactors()

metrics_evaluator = UserCumulativeInteractionMetricsEvaluator(None, metrics_classes)

# evaluation_policy = ir.get_interactors_evaluation_policy()
evaluation_policy_name = settings['defaults']['interactors_evaluation_policy']
evaluation_policy_parameters = settings['evaluation_policies_parameters'][evaluation_policy_name]
evaluation_policy=eval('lib.evaluation_policies.'+evaluation_policy_name)(**evaluation_policy_parameters)

nums_interactions_to_show = list(map(int,args.i))


def generate_table_spec():
    res = '|'
    for i in range(1 + len(nums_interactions_to_show) *
                   len(datasets_preprocessors)):
        res += 'c'
        if i % (len(nums_interactions_to_show)) == 0:
            res += '|'
    return res


rtex_header = r"""
\documentclass{standalone}
%%\usepackage[landscape, paperwidth=15cm, paperheight=30cm, margin=0mm]{geometry}
\usepackage{multirow}
\usepackage{color, colortbl}
\usepackage{xcolor, soul}
\usepackage{amssymb}
\definecolor{Gray}{gray}{0.9}
\definecolor{StrongGray}{gray}{0.7}
\begin{document}
\begin{tabular}{%s}
\hline
\rowcolor{StrongGray}
Dataset & %s \\""" % (generate_table_spec(), ' & '.join([
    r"\multicolumn{%d}{c|}{%s}" % (len(nums_interactions_to_show), i['name'])
    for i in datasets_preprocessors
]))
rtex_footer = r"""
\end{tabular}
\end{document}
"""
rtex = ""

datasets_metrics_values = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list)))

datasets_metrics_users_values = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list)))

for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)

    for metric_class_name in metrics_classes_names:
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
                metric_class_name][itr_class.__name__].extend(
                    [np.mean(list(metric_values[i].values())) for i in range(len(nums_interactions_to_show))])
            datasets_metrics_users_values[dataset_preprocessor['name']][
                metric_class_name][itr_class.__name__].extend(
                    np.array([list(metric_values[i].values()) for i in range(len(nums_interactions_to_show))]))

            # print(datasets_metrics_values[dataset_preprocessor['name']][
                # metric_class_name][itr_class.__name__])
            # print(datasets_metrics_users_values[dataset_preprocessor['name']][
                # metric_class_name][itr_class.__name__])


utility_scores = defaultdict(
        lambda: defaultdict(lambda: defaultdict(
        lambda:dict())))
method_utility_scores = defaultdict(
        lambda: defaultdict(lambda: dict))
for num_interaction in range(len(nums_interactions_to_show)):
    for dataset_preprocessor in datasets_preprocessors:
        dm.initialize_engines(dataset_preprocessor)
        for metric_class_name in metrics_classes_names:
            for itr_class in interactors_classes:
                metric_max_value = np.max(list(map(lambda x:x[num_interaction],datasets_metrics_values[dataset_preprocessor['name']][
                    metric_class_name].values())))
                metric_min_value = np.min(list(map(lambda x:x[num_interaction],datasets_metrics_values[dataset_preprocessor['name']][
                    metric_class_name].values())))
                metric_value = datasets_metrics_values[dataset_preprocessor['name']][
                                metric_class_name][itr_class.__name__][num_interaction]
                utility_scores[dataset_preprocessor['name']][metric_class_name][itr_class.__name__][num_interaction] =\
                        (metric_value - metric_min_value)/(metric_max_value-metric_min_value)
                # print(f"({metric_value} - {metric_min_value})/({metric_max_value}-{metric_min_value})")

# methods_utilities = defaultdict(
        # lambda: defaultdict(lambda: dict()))
for num_interaction in range(len(nums_interactions_to_show)):
    for dataset_preprocessor in datasets_preprocessors:
        dm.initialize_engines(dataset_preprocessor)
        # for metric_class_name in map(lambda x: x.__name__, metrics_classes):
        for itr_class in interactors_classes:
            # print([utility_scores[dataset_preprocessor['name']][metric_class_name][itr_class.__name__][num_interaction]*metrics_weights[metric_class_name] for metric_class_name in metrics_classes_names])
            us= [utility_scores[dataset_preprocessor['name']][metric_class_name][itr_class.__name__][num_interaction]*metrics_weights[metric_class_name] for metric_class_name in metrics_classes_names]
            # print(us)
            maut = np.sum(us)
            # datasets_metrics_values[dataset_preprocessor['name']]['MAUT'][itr_class.__name__] = []
            # datasets_metrics_users_values[dataset_preprocessor['name']]['MAUT'][itr_class.__name__] = []
            datasets_metrics_values[dataset_preprocessor['name']]['MAUT'][itr_class.__name__].append(maut)
            # if datasets_metrics_users_values[dataset_preprocessor['name']]['MAUT'][itr_class.__name__] == None:
            datasets_metrics_users_values[dataset_preprocessor['name']]['MAUT'][itr_class.__name__].append(np.array([maut]*100))

            # print(num_interaction,us,maut,datasets_metrics_values[dataset_preprocessor['name']]['MAUT'][itr_class.__name__])
            # print('maut',maut,datasets_metrics_values[dataset_preprocessor['name']]['MAUT'][itr_class.__name__],datasets_metrics_users_values[dataset_preprocessor['name']]['MAUT'][itr_class.__name__])
            
if args.dump:
    # with open('datasets_metrics_values.pickle','wb') as f:
        # pickle.dump(datasets_metrics_values,f)
    with open('datasets_metrics_values.pickle','wb') as f:
        pickle.dump(json.loads(json.dumps(datasets_metrics_values)),f)
        # f.write(str(methods_users_hits))
# print(datasets_metrics_values['Yahoo Music']['MAUT'])


metrics_classes_names.append('MAUT')
metrics_names.append('MAUT')

datasets_metrics_gain = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: ['']*len(nums_interactions_to_show))))

datasets_metrics_best = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda:[False]*len(nums_interactions_to_show))))
bullet_str = r'\textcolor[rgb]{0.7,0.7,0.0}{$\bullet$}'
triangle_up_str = r'\textcolor[rgb]{00,0.45,0.10}{$\blacktriangle$}'
triangle_down_str = r'\textcolor[rgb]{0.7,00,00}{$\blacktriangledown$}'

if args.type == 'pairs':
    pool_of_methods_to_compare = [(interactors_classes[i].__name__,interactors_classes[i+1].__name__) for i in range(0,len(interactors_classes)-1,2)]
    print(pool_of_methods_to_compare)
else:
    pool_of_methods_to_compare = [[interactors_classes[i].__name__ for i in range(len(interactors_classes)-1)]]
for dataset_preprocessor in datasets_preprocessors:
    for metric_class_name in metrics_classes_names:
        for i, num in enumerate(nums_interactions_to_show):
            for methods in pool_of_methods_to_compare:
                datasets_metrics_values_tmp = copy.deepcopy(datasets_metrics_values)
                methods_set = set(methods)
                for k1, v1 in datasets_metrics_values.items():
                    for k2, v2 in v1.items():
                        for k3, v3 in v2.items():
                            if k3 not in methods_set:
                                del datasets_metrics_values_tmp[k1][k2][k3]
                        # print(datasets_metrics_values_tmp[k1][k2])
                # datasets_metrics_values_tmp =datasets_metrics_values
                datasets_metrics_best[dataset_preprocessor['name']][metric_class_name][max(datasets_metrics_values_tmp[dataset_preprocessor['name']][
                        metric_class_name].items(),key=lambda x: x[1][i])[0]][i] = True
                if args.r == 'lastmethod':
                    best_itr = methods[-1]
                elif args.r != None:
                    best_itr = args.r
                else:
                    best_itr = max(datasets_metrics_values_tmp[dataset_preprocessor['name']][
                        metric_class_name].items(),key=lambda x: x[1][i])[0]
                best_itr_vals = datasets_metrics_values_tmp[dataset_preprocessor['name']][
                        metric_class_name].pop(best_itr)
                best_itr_val = best_itr_vals[i]
                second_best_itr = max(datasets_metrics_values_tmp[dataset_preprocessor['name']][
                    metric_class_name].items(),key=lambda x: x[1][i])[0]
                second_best_itr_vals = datasets_metrics_values_tmp[dataset_preprocessor['name']][
                        metric_class_name][second_best_itr]
                second_best_itr_val = second_best_itr_vals[i]
                # come back with value in dict
                datasets_metrics_values_tmp[dataset_preprocessor['name']][

                        metric_class_name][best_itr] = best_itr_vals

                best_itr_users_val = datasets_metrics_users_values[dataset_preprocessor['name']][
                    metric_class_name][best_itr][i]
                second_best_itr_users_val = datasets_metrics_users_values[dataset_preprocessor['name']][
                    metric_class_name][second_best_itr][i]

                try:
                    statistic, pvalue = scipy.stats.wilcoxon(
                            best_itr_users_val,
                            second_best_itr_users_val,
                            )
                except:
                    print("Wilcoxon error")
                    datasets_metrics_gain[dataset_preprocessor['name']][metric_class_name][best_itr][i]=bullet_str

                if pvalue > 0.05:
                    datasets_metrics_gain[dataset_preprocessor['name']][metric_class_name][best_itr][i]=bullet_str
                else:
                    # print(best_itr,best_itr_val,second_best_itr,second_best_itr_val,methods)
                    if best_itr_val < second_best_itr_val:
                        datasets_metrics_gain[dataset_preprocessor['name']][metric_class_name][best_itr][i]=triangle_down_str
                    elif best_itr_val > second_best_itr_val:
                        datasets_metrics_gain[dataset_preprocessor['name']][metric_class_name][best_itr][i]=triangle_up_str
                    else:
                        datasets_metrics_gain[dataset_preprocessor['name']][metric_class_name][best_itr][i]=bullet_str


for metric_name, metric_class_name in zip(
        metrics_names, metrics_classes_names):
    rtex += utils.generate_metric_interactions_header(nums_interactions_to_show,len(datasets_preprocessors),metric_name)
    for itr_class in interactors_classes:
        rtex += "%s & " % (ir.get_interactor_name(itr_class.__name__))
        rtex += ' & '.join([
            ' & '.join(
                map(
                    lambda x,y,z: (r"\textbf{" if z else "") + f"{x:.3f}{y}" + (r"}" if z else ""),
                    datasets_metrics_values[dataset_preprocessor['name']]
                    [metric_class_name][itr_class.__name__],
                    datasets_metrics_gain[dataset_preprocessor['name']][metric_class_name][itr_class.__name__],
                    datasets_metrics_best[dataset_preprocessor['name']][metric_class_name][itr_class.__name__]
                    ))
            for dataset_preprocessor in datasets_preprocessors
        ])
        rtex += r'\\\hline' + '\n'

res = rtex_header + rtex + rtex_footer

tmp = '_'.join([
    dataset_preprocessor['name']
    for dataset_preprocessor in datasets_preprocessors
])
open(os.path.join(DirectoryDependent().DIRS['tex'], f'table_{tmp}.tex'),
     'w+').write(res)
os.system(
    f"pdflatex -output-directory=\"{DirectoryDependent().DIRS['pdf']}\" \"{os.path.join(DirectoryDependent().DIRS['tex'],f'table_{tmp}.tex')}\""
)
