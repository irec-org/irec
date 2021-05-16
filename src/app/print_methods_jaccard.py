from os.path import dirname, realpath, sep, pardir
import pickle
import os
import sys
import json
import utils
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
import copy
import pandas as pd
import seaborn as sn
import lib.evaluation_policies
import scipy
import interactors
import mf
from lib.utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from lib.utils.DatasetManager import DatasetManager
import yaml
from metrics import CumulativeInteractionMetricsEvaluator, UserCumulativeInteractionMetricsEvaluator, UsersCoverage
from lib.utils.dataset import Dataset
from lib.utils.PersistentDataManager import PersistentDataManager
from lib.utils.InteractorCache import InteractorCache
import metrics
import matplotlib.pyplot as plt
from lib.utils.DirectoryDependent import DirectoryDependent
from cycler import cycler
from collections import defaultdict
import argparse
import lib.utils.utils
parser = argparse.ArgumentParser()
parser.add_argument('-r', type=str, default=None)
parser.add_argument('-i', default=[5, 10, 20, 50, 100], nargs='*')
parser.add_argument('-m', nargs='*')
parser.add_argument('-b', nargs='*')
parser.add_argument('--dump', default=False, action='store_true')
parser.add_argument('--type', default='items_jac')
settings = utils.load_settings()
utils.load_settings_to_parser(settings,parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings,args)

plt.rcParams['axes.prop_cycle'] = cycler(color='krbgmyc')
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 15

# metrics_classes = [metrics.Hits, metrics.Recall]
metrics_classes_names = [args.type]
metrics_names = [args.type]
# metrics_weights = {'Hits': 0.3,'Recall':0.3,'EPC':0.16666,'UsersCoverage':0.16666,'ILD':0.16666}
# metrics_weights = {'Hits': 0.25,'Recall':0.25,'EPC':0.125,'UsersCoverage':0.125,'ILD':0.125,'GiniCoefficientInv':0.125}
# metrics_weights ={i: 1/len(metrics_classes_names) for i in metrics_classes_names}

interactors_classes_names_to_names = {
    k: v['name'] for k, v in settings['interactors_general_settings'].items()
}

dm = DatasetManager()
datasets_preprocessors = [settings['datasets_preprocessors_parameters'][base] for base in args.b]
interactors_classes = [
    eval('lib.interactors.' + interactor) for interactor in args.m
]

# ir = InteractorRunner(dm, interactors_general_settings,
# interactors_preprocessor_parameters,
# evaluation_policies_parameters)
# interactors_classes = ir.select_interactors()

# metrics_evaluator = UserCumulativeInteractionMetricsEvaluator(
    # None, metrics_classes)

evaluation_policy_name = settings['defaults']['interactors_evaluation_policy']
evaluation_policy_parameters = settings['evaluation_policies_parameters'][evaluation_policy_name]
evaluation_policy=eval('lib.evaluation_policies.'+evaluation_policy_name)(**evaluation_policy_parameters)

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
datasets_interactors_items_recommended=defaultdict(
    lambda: defaultdict(list))

for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)
    for itr_class in interactors_classes:
        # itr = ir.create_interactor(itr_class)
        itr = utils.create_interactor(itr_class,dataset_preprocessor['name'],settings)
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
    if args.type=='users':
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
        itr_1 = utils.create_interactor(itr_class_1,dataset_preprocessor['name'],settings)
        for jj, itr_class_2 in enumerate(interactors_classes):
            if ii > jj:
                itr_2 = utils.create_interactor(itr_class_2,dataset_preprocessor['name'],settings)
                name = interactors_classes_names_to_names[itr_class_1.__name__]+r' $\times $ '+interactors_classes_names_to_names[itr_class_2.__name__]
                itr_1_recs = datasets_interactors_items_recommended[dataset_preprocessor['name']][itr_class_1.__name__]
                itr_2_recs = datasets_interactors_items_recommended[dataset_preprocessor['name']][itr_class_2.__name__]
                vals = []
                if args.type == 'Users Jaccard Similarity':
                    for i in nums_interactions_to_show:
                        x = set()
                        for uid,items in itr_1_recs.items():
                            x |= set(items[:i])
                        y = set()
                        for uid,items in itr_2_recs.items():
                            y |= set(items[:i])
                        vals.append(len(x.intersection(y))/len(x | y))
                elif args.type =='Items Jaccard Similarity':
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
                elif args.type == 'Users-Items Similarity':
                    for i in nums_interactions_to_show:
                        x = []
                        for uid,items in itr_1_recs.items():
                            sample_rec_items = list(set(items[:i]).intersection(set(itr_2_recs[uid][:i])))
                            x.append(len(sample_rec_items)/len(items[:i]))
                            # if np.sum(ground_truth_consumption_matrix[uid,items[:i]] >= 4)>0:
                                # x.add(uid)
                        vals.append(np.mean(x))


                for iii, nits in enumerate(nums_interactions_to_show):
                    dfs[nits][interactors_classes_names_to_names[itr_class_1.__name__]][interactors_classes_names_to_names[itr_class_2.__name__]] = vals[iii]
                    dfs[nits][interactors_classes_names_to_names[itr_class_2.__name__]][interactors_classes_names_to_names[itr_class_1.__name__]] = vals[iii]

                datasets_metrics_values[dataset_preprocessor['name']][args.type][name].extend(vals)
                methods_names.add(name)

                # print(vals)
    for iii, nits in enumerate(nums_interactions_to_show):
        fig = plt.figure(figsize = (16,16))
        print(dfs[nits])
        sns_plot=sn.heatmap(dfs[nits], annot=True,cmap="Blues",vmin=0,vmax=1)
        sns_plot.set_title(f"T={nits} {dataset_preprocessor['name']}")
        file_name=os.path.join(DirectoryDependent().DIRS['img'],f'{dataset_preprocessor["name"]}_{args.type}',f'cm_{evaluation_policy.num_interactions}_{evaluation_policy.interaction_size}',f'cm_{nits}.png')
        lib.utils.utils.create_path_to_file(file_name)
        fig.savefig(file_name)

if args.dump:
    with open('datasets_metrics_values.pickle', 'wb') as f:
        pickle.dump(json.loads(json.dumps(datasets_metrics_values)), f)

for metric_name, metric_class_name in zip(
        metrics_names, metrics_classes_names):
    rtex += utils.generate_metric_interactions_header(nums_interactions_to_show,len(datasets_preprocessors),metric_name)
    for method_name in methods_names:
        rtex += "{} & ".format(method_name)
        bases_values = []
        for dataset_preprocessor in datasets_preprocessors:
            bases_values.append(' & '.join(
                map(
                    lambda x: f"{x:.3f}",
                    datasets_metrics_values[dataset_preprocessor['name']]
                    [metric_class_name][method_name],
                    )))
            print(bases_values)
        rtex+= ' & '.join(bases_values)
        rtex+=r'\\\hline'+'\n'

res = rtex_header + rtex + rtex_footer

tmp = '_'.join([
    dataset_preprocessor['name']
    for dataset_preprocessor in datasets_preprocessors
])

file_name_suffix = os.path.join(f'table_similarity_{args.type}_{evaluation_policy.num_interactions}_{evaluation_policy.interaction_size}_{tmp}.tex')
file_name =os.path.join(DirectoryDependent().DIRS['tex'],file_name_suffix)
open(file_name,
     'w+').write(res)
os.system(
    f"pdflatex -output-directory=\"{DirectoryDependent().DIRS['pdf']}\" \"{file_name}\""
)
