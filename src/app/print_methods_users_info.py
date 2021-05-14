from os.path import dirname, realpath, sep, pardir
import pickle
import os
import sys
import json
sys.path.append(dirname(realpath(__file__)) + sep + pardir)

import inquirer
import copy
import pandas as pd
import seaborn as sn
import scipy
import lib.interactors
import mf
from lib.utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from lib.utils.DatasetManager import DatasetManager
import yaml
from metric import CumulativeInteractionMetricsEvaluator, UserCumulativeInteractionMetricsEvaluator, UsersCoverage
from lib.utils.dataset import Dataset
from lib.utils.PersistentDataManager import PersistentDataManager
from lib.utils.InteractorCache import InteractorCache
import metric
import matplotlib.pyplot as plt
from lib.utils.DirectoryDependent import DirectoryDependent
from cycler import cycler
from collections import defaultdict
import argparse
import lib.utils.utils
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-r', type=str, default=None)
parser.add_argument('-i', default=[5, 10, 20, 50, 100], nargs='*')
parser.add_argument('-m', nargs='*')
parser.add_argument('-b', nargs='*')
settings = utils.load_settings()
utils.load_settings_to_parser(settings,parser)

args = parser.parse_args()
print(args)
settings = utils.sync_settings_from_args(settings,args)

interactors_classes_names_to_names = {
    k: v['name'] for k, v in settings['interactors_general_settings'].items()
}

dm = DatasetManager()
datasets_preprocessors = [settings['datasets_preprocessors_parameters'][base] for base in args.b]

interactors_classes = [
    eval('interactors.' + interactor) for interactor in args.m
]

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
    utils.generate_table_spec(nums_interactions_to_show, len(datasets_preprocessors)),
    utils.generate_datasets_line(nums_interactions_to_show,
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

methods_names= set()
for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)
    if args.users:
        dm.load()
        data = np.vstack(
            (dm.dataset_preprocessed[0].data, dm.dataset_preprocessed[1].data))

        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        ground_truth_dataset= dataset
        ground_truth_consumption_matrix = scipy.sparse.csr_matrix(
            (ground_truth_dataset.data[:, 2],
             (ground_truth_dataset.data[:, 0],
              ground_truth_dataset.data[:, 1])),
            (ground_truth_dataset.num_total_users,
             ground_truth_dataset.num_total_items))
    dfs = dict()
    for nits in nums_interactions_to_show:
        array = np.array([[1]*len(interactors_classes)]*len(interactors_classes),dtype=float)
        dfs[nits] = pd.DataFrame(array, index = [interactors_classes_names_to_names[i.__name__] for i in interactors_classes],
                          columns = [interactors_classes_names_to_names[i.__name__] for i in interactors_classes])
    for ii, itr_class_1 in enumerate(interactors_classes):
        itr_1 = ir.create_interactor(itr_class_1)
        for jj, itr_class_2 in enumerate(interactors_classes):
            if ii > jj:
                itr_2 = ir.create_interactor(itr_class_1)
                name = interactors_classes_names_to_names[itr_class_1.__name__]+r' $\times $ '+interactors_classes_names_to_names[itr_class_2.__name__]
                itr_1_recs = datasets_interactors_items_recommended[dataset_preprocessor['name']][itr_class_1.__name__]
                itr_2_recs = datasets_interactors_items_recommended[dataset_preprocessor['name']][itr_class_2.__name__]
                vals = []
                if args.users == False:
                    for i in nums_interactions_to_show:
                        x = set()
                        for uid,items in itr_1_recs.items():
                            x |= set(items[:i])
                        y = set()
                        for uid,items in itr_2_recs.items():
                            y |= set(items[:i])
                        vals.append(len(x.intersection(y))/len(x | y))
                else:
                    for i in nums_interactions_to_show:
                        x = set()
                        for uid,items in itr_1_recs.items():
                            if np.sum(ground_truth_consumption_matrix[uid,items[:i]] >= 4)>0:
                                x.add(uid)
                        y = set()
                        for uid,items in itr_2_recs.items():
                            if np.sum(ground_truth_consumption_matrix[uid,items[:i]] >= 4)>0:
                                y.add(uid)
                        vals.append(len(x.intersection(y))/len(x | y))


                for iii, nits in enumerate(nums_interactions_to_show):
                    dfs[nits][interactors_classes_names_to_names[itr_class_1.__name__]][interactors_classes_names_to_names[itr_class_2.__name__]] = vals[iii]
                    dfs[nits][interactors_classes_names_to_names[itr_class_2.__name__]][interactors_classes_names_to_names[itr_class_1.__name__]] = vals[iii]

                datasets_metrics_values[dataset_preprocessor['name']]['Jaccard Similarity'][name].extend(vals)
                methods_names.add(name)

                # print(vals)
