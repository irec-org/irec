from os.path import dirname, realpath, sep, pardir
import pickle
import os
import sys
import json
from util import *
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
parser.add_argument('-i', default=[5, 10, 20, 50, 100], nargs='*')
parser.add_argument('-m', nargs='*')
parser.add_argument('-b', nargs='*')
parser.add_argument('--dump', default=False, action='store_true')
args = parser.parse_args()

plt.rcParams['axes.prop_cycle'] = cycler(color='krbgmyc')
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 15

# metrics_classes = [metric.Hits, metric.Recall]
metrics_classes_names = ['Jaccard Similarity']
metrics_names = ['Jaccard Similarity']
# metrics_weights = {'Hits': 0.3,'Recall':0.3,'EPC':0.16666,'UsersCoverage':0.16666,'ILD':0.16666}
# metrics_weights = {'Hits': 0.25,'Recall':0.25,'EPC':0.125,'UsersCoverage':0.125,'ILD':0.125,'GiniCoefficientInv':0.125}
# metrics_weights ={i: 1/len(metrics_classes_names) for i in metrics_classes_names}

with open("settings" + sep + "datasets_preprocessors_parameters.yaml") as f:
    loader = yaml.SafeLoader
    datasets_preprocessors = yaml.load(f, Loader=loader)

    datasets_preprocessors = {
        setting['name']: setting for setting in datasets_preprocessors
    }
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
interactors_classes = [
    eval('interactors.' + interactor) for interactor in args.m
]

# ir = InteractorRunner(dm, interactors_general_settings,
# interactors_preprocessor_paramaters,
# evaluation_policies_parameters)
# interactors_classes = ir.select_interactors()

# metrics_evaluator = UserCumulativeInteractionMetricsEvaluator(
    # None, metrics_classes)

evaluation_policy = ir.get_interactors_evaluation_policy()

nums_interactions_to_show = list(map(int, args.i))

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
Dataset & %s \\""" % (
    generate_table_spec(nums_interactions_to_show, len(datasets_preprocessors)),
    generate_datasets_line(nums_interactions_to_show,
                           [i['name'] for i in datasets_preprocessors]))
rtex_footer = r"""
\end{tabular}
\end{document}
"""
rtex = ""

datasets_metrics_values = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list)))
datasets_interactors_items_recommended = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list)))

for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)
    for itr_class in interactors_classes:
        itr = ir.create_interactor(itr_class)
        pdm = PersistentDataManager(directory='results')
        history_items_recommended = pdm.load(InteractorCache().get_id(
            dm, evaluation_policy, itr))
        users_items_recommended = defaultdict(list)
        for i in range(len(history_items_recommended)):
            uid = history_items_recommended[i][0]
            iid = history_items_recommended[i][1]
            users_items_recommended[uid].append(iid)

        datasets_interactors_items_recommended[dataset_preprocessor['name']][itr_class.__name__] = users_items_recommended

for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)
    for ii, itr_class_1 in enumerate(interactors_classes):
        itr_1 = ir.create_interactor(itr_class_1)
        for jj, itr_class_2 in enumerate(interactors_classes):
            if ii > jj:
                itr_2 = ir.create_interactor(itr_class_1)
                name = interactors_classes_names_to_names[itr_class_1.__name__]+r' $\times $ '+interactors_classes_names_to_names[itr_class_2.__name__]
                itr_1_recs = datasets_interactors_items_recommended[dataset_preprocessor['name']][itr_class_1.__name__]
                itr_2_recs = datasets_interactors_items_recommended[dataset_preprocessor['name']][itr_class_2.__name__]
                vals = []
                for i in nums_interactions_to_show:
                    x = set()
                    for uid,items in itr_1_recs.items():
                        x |= set(items[:i])
                    y = set()
                    for uid,items in itr_2_recs.items():
                        y |= set(items[:i])
                    vals.append(len(x.intersection(y))/len(x | y))

                datasets_metrics_values[dataset_preprocessor['name']]['Jaccard Similarity'][name].extend(vals)
                # print(vals)

# print(datasets_metrics_values)
utility_scores = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: dict())))
method_utility_scores = defaultdict(lambda: defaultdict(lambda: dict))
# for num_interaction in range(len(nums_interactions_to_show)):
    # for dataset_preprocessor in datasets_preprocessors:
        # dm.initialize_engines(dataset_preprocessor)
        # for metric_class_name in metrics_classes_names:
            # for itr_class in interactors_classes:
                # metric_max_value = np.max(
                    # list(
                        # map(
                            # lambda x: x[num_interaction],
                            # datasets_metrics_values[dataset_preprocessor[
                                # 'name']][metric_class_name].values())))
                # metric_min_value = np.min(
                    # list(
                        # map(
                            # lambda x: x[num_interaction],
                            # datasets_metrics_values[dataset_preprocessor[
                                # 'name']][metric_class_name].values())))
                # metric_value = datasets_metrics_values[
                    # dataset_preprocessor['name']][metric_class_name][
                        # itr_class.__name__][num_interaction]
                # utility_scores[dataset_preprocessor['name']][metric_class_name][itr_class.__name__][num_interaction] =\
                        # (metric_value - metric_min_value)/(metric_max_value-metric_min_value)
                # print(f"({metric_value} - {metric_min_value})/({metric_max_value}-{metric_min_value})")

# methods_utilities = defaultdict(
# lambda: defaultdict(lambda: dict()))

if args.dump:
    # with open('datasets_metrics_values.pickle','wb') as f:
    # pickle.dump(datasets_metrics_values,f)
    with open('datasets_metrics_values.pickle', 'wb') as f:
        pickle.dump(json.loads(json.dumps(datasets_metrics_values)), f)
        # f.write(str(methods_users_hits))
# print(datasets_metrics_values['Yahoo Music']['MAUT'])

# metrics_classes_names.append('MAUT')
# metrics_names.append('MAUT')

datasets_metrics_gain = defaultdict(lambda: defaultdict(lambda: defaultdict(
    lambda: [''] * len(nums_interactions_to_show))))

datasets_metrics_best = defaultdict(lambda: defaultdict(lambda: defaultdict(
    lambda: [False] * len(nums_interactions_to_show))))
bullet_str = r'\textcolor[rgb]{0.7,0.7,0.0}{$\bullet$}'
triangle_up_str = r'\textcolor[rgb]{00,0.45,0.10}{$\blacktriangle$}'
triangle_down_str = r'\textcolor[rgb]{0.7,00,00}{$\blacktriangledown$}'
for dataset_preprocessor in datasets_preprocessors:
    for metric_class_name in metrics_classes_names:
        for i, num in enumerate(nums_interactions_to_show):
            # for itr_class in interactors_classes:
            datasets_metrics_best[
                dataset_preprocessor['name']][metric_class_name][max(
                    datasets_metrics_values[dataset_preprocessor['name']]
                    [metric_class_name].items(),
                    key=lambda x: x[1][i])[0]][i] = True
            if args.r != None:
                best_itr = args.r
            else:
                best_itr = max(datasets_metrics_values[
                    dataset_preprocessor['name']][metric_class_name].items(),
                               key=lambda x: x[1][i])[0]
            best_itr_vals = datasets_metrics_values[
                dataset_preprocessor['name']][metric_class_name].pop(best_itr)
            best_itr_val = best_itr_vals[i]
            # print(datasets_metrics_values[
                # ])
            # print(dataset_preprocessor['name'],metric_class_name)
            # print(dataset_preprocessor['name'],metric_class_name,datasets_metrics_values[
                # dataset_preprocessor['name']])
            # print(datasets_metrics_values)
            second_best_itr = max(datasets_metrics_values[
                dataset_preprocessor['name']][metric_class_name].items(),
                                  key=lambda x: x[1][i])[0]
            second_best_itr_vals = datasets_metrics_values[dataset_preprocessor[
                'name']][metric_class_name][second_best_itr]
            second_best_itr_val = second_best_itr_vals[i]
            # come back with value in dict
            datasets_metrics_values[dataset_preprocessor['name']][
                metric_class_name][best_itr] = best_itr_vals

            best_itr_users_val = datasets_metrics_users_values[
                dataset_preprocessor['name']][metric_class_name][best_itr][i]
            second_best_itr_users_val = datasets_metrics_users_values[
                dataset_preprocessor['name']][metric_class_name][
                    second_best_itr][i]

            try:
                statistic, pvalue = scipy.stats.wilcoxon(
                    best_itr_users_val,
                    second_best_itr_users_val,
                )
            except:
                print("Wilcoxon error")
                datasets_metrics_gain[dataset_preprocessor['name']][
                    metric_class_name][best_itr][i] = bullet_str

            if pvalue > 0.05:
                datasets_metrics_gain[dataset_preprocessor['name']][
                    metric_class_name][best_itr][i] = bullet_str
            else:
                if best_itr_val < second_best_itr_val:
                    datasets_metrics_gain[dataset_preprocessor['name']][
                        metric_class_name][best_itr][i] = triangle_down_str
                elif best_itr_val > second_best_itr_val:
                    datasets_metrics_gain[dataset_preprocessor['name']][
                        metric_class_name][best_itr][i] = triangle_up_str
                else:
                    datasets_metrics_gain[dataset_preprocessor['name']][
                        metric_class_name][best_itr][i] = bullet_str

for metric_name, metric_class_name in zip(metrics_names, metrics_classes_names):
    rtex += generate_metric_interactions_header(nums_interactions_to_show,
                                                len(datasets_preprocessors),
                                                metric_name)
    for itr_class in interactors_classes:
        rtex += "%s & " % (ir.get_interactor_name(itr_class.__name__))
        rtex += ' & '.join([
            ' & '.join(
                map(
                    lambda x, y, z: (r"\textbf{" if z else "") + f"{x:.3f}{y}" +
                    (r"}" if z else ""),
                    datasets_metrics_values[dataset_preprocessor['name']]
                    [metric_class_name][itr_class.__name__],
                    datasets_metrics_gain[dataset_preprocessor['name']]
                    [metric_class_name][itr_class.__name__],
                    datasets_metrics_best[dataset_preprocessor['name']]
                    [metric_class_name][itr_class.__name__]))
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
