from os.path import dirname, realpath, sep, pardir
import pickle
import os
import matplotlib.ticker as mtick
import sys
import json
sys.path.append(dirname(realpath(__file__)) + sep + pardir)
from lib.metrics import InteractionMetricsEvaluator, CumulativeMetricsEvaluator, CumulativeInteractionMetricsEvaluator, UserCumulativeInteractionMetricsEvaluator
import inquirer
from tabulate import tabulate
import lib.metrics
import copy
import pandas as pd
import seaborn as sn
import scipy
import scipy.stats
import lib.interactors
import lib.evaluation_policies
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
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-r', type=str, default=None)
parser.add_argument('-i', default=[5, 10, 20, 50, 100], nargs='*')
parser.add_argument('-m', nargs='*')
parser.add_argument('-b', nargs='*')
parser.add_argument('-e',default=['Hits','Recall'], nargs='*')
settings = utils.load_settings()
utils.load_settings_to_parser(settings,parser)

args = parser.parse_args()
settings = utils.sync_settings_from_args(settings,args)

nums_interactions_to_show=args.i
interactors_classes_names_to_names = {
    k: v['name'] for k, v in settings['interactors_general_settings'].items()
}

dm = DatasetManager()
datasets_preprocessors = [settings['datasets_preprocessors_parameters'][base] for base in args.b]

interactors_classes = [
    eval('lib.interactors.' + interactor) for interactor in args.m
]

metrics_classes_names = args.e
metrics_classes = [eval('lib.metrics.'+ metric) for metric in args.e]
metrics_evaluator = UserCumulativeInteractionMetricsEvaluator(None, metrics_classes)


evaluation_policy_name = settings['defaults']['interactors_evaluation_policy']
evaluation_policy_parameters = settings['evaluation_policies_parameters'][evaluation_policy_name]
evaluation_policy=eval('lib.evaluation_policies.'+evaluation_policy_name)(**evaluation_policy_parameters)
# datasets_interactors_items_recommended = defaultdict(
    # lambda: defaultdict(lambda: defaultdict(list)))


# for dataset_preprocessor in datasets_preprocessors:
    # dm.initialize_engines(dataset_preprocessor)
    # for itr_class in interactors_classes:
        # itr = itr_class(**settings['interactors_preprocessor_paramaters'][dataset_preprocessor['name']][itr_class.__name__]['parameters'])
        # pdm = PersistentDataManager(directory='results')
        # history_items_recommended = pdm.load(InteractorCache().get_id(
            # dm, evaluation_policy, itr))
        # users_items_recommended = defaultdict(list)
        # for i in range(len(history_items_recommended)):
            # uid = history_items_recommended[i][0]
            # iid = history_items_recommended[i][1]
            # users_items_recommended[uid].append(iid)
        # # uir = defaultdict(list)
        # # for k,v in users_items_recommended.items():
            # # uir[k] = v[:10]

        # datasets_interactors_items_recommended[dataset_preprocessor['name']][itr_class.__name__] = users_items_recommended
        # datasets_interactors_items_recommended[dataset_preprocessor['name']][itr_class.__name__] = uir

datasets_metrics_values = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list)))

datasets_metrics_users_values = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list)))
for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)

    for metric_class_name in metrics_classes_names:
        for itr_class in interactors_classes:
            itr = itr_class(**settings['interactors_preprocessor_paramaters'][dataset_preprocessor['name']][itr_class.__name__]['parameters'])
            pdm = PersistentDataManager(directory='results')

            metrics_pdm = PersistentDataManager(directory='metrics')
            metric_values = metrics_pdm.load(
                os.path.join(
                    InteractorCache().get_id(dm, evaluation_policy, itr),
                    metrics_evaluator.get_id(), metric_class_name))
            datasets_metrics_values[dataset_preprocessor['name']][
                metric_class_name][itr_class.__name__].extend(
                    [np.mean(metric_values[i]) for i in range(len(nums_interactions_to_show))])
            datasets_metrics_users_values[dataset_preprocessor['name']][
                metric_class_name][itr_class.__name__].extend(
                    np.array([metric_values[i] for i in range(len(nums_interactions_to_show))]))

methods_names= set()
results = [['']*(len(metrics_classes_names)*len(datasets_preprocessors)+2)]*(2+len(interactors_classes)*4)
plt.style.use('seaborn-dark')
fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(7,7))
axs[1,0].set_ylabel('Mean Rating')
axs[1,0].set_xlabel('Users Rank (%)')
axs[1,1].set_ylabel('Rating Variance')
axs[1,1].set_xlabel('Users Rank (%)')
axs[0,0].set_xlabel('Items Rank (%)')
axs[0,0].set_ylabel('Popularity (%)')
axs[0,1].set_xlabel('Users Rank (%)')
axs[0,1].set_ylabel('Consumption History (%)')
for hh, dataset_preprocessor in enumerate(datasets_preprocessors):
    dm.initialize_engines(dataset_preprocessor)
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

    df = pd.DataFrame(ground_truth_dataset.data)
    df = df.sort(0)
    # print(df.head())
    # print(df.groupby(0).mean().head())
    ds_data = dict()
    upercent= np.array(list(range(ground_truth_dataset.num_total_users)))
    upercent = upercent/np.max(upercent)*100
    ipercent= np.array(list(range(ground_truth_dataset.num_total_items)))
    ipercent = ipercent/np.max(ipercent)*100
    y=np.sort(np.array(df.groupby(0).mean()[2]))[::-1]
    ds_ratingsm = y
    ds_data['Ratings Mean']=y
    axs[1,0].plot(upercent,y,label=dataset_preprocessor['name'])
    axs[1,0].xaxis.set_major_formatter(mtick.PercentFormatter())
    y=np.sort(np.array(df.groupby(0).var()[2]))[::-1]
    ds_ratingsv=y
    ds_data['Ratings Variance']=y
    axs[1,1].plot(upercent,y,label=dataset_preprocessor['name'])
    axs[1,1].xaxis.set_major_formatter(mtick.PercentFormatter())
    y=100*np.sort(np.array(df.groupby(0).count()[2]))[::-1]/ground_truth_dataset.num_total_items
    ds_history = y
    ds_data['History']=y
    axs[0,1].plot(upercent,y,label=dataset_preprocessor['name'])
    axs[0,1].xaxis.set_major_formatter(mtick.PercentFormatter())
    axs[0,1].yaxis.set_major_formatter(mtick.PercentFormatter())
    y=100*np.sort(np.array(df.groupby(1).count()[2]))[::-1]/ground_truth_dataset.num_total_users
    ds_popularity = y
    # ds_data['Popularity']=y
    axs[0,0].plot(ipercent,y,label=dataset_preprocessor['name'])
    axs[0,0].xaxis.set_major_formatter(mtick.PercentFormatter())
    axs[0,0].yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # np.sort(ground_truth_consumption_matrix.toarray())
    for ii, itr_class in enumerate(interactors_classes):
        itr = itr_class(**settings['interactors_preprocessor_paramaters'][dataset_preprocessor['name']][itr_class.__name__]['parameters'])
        # itr_recs = datasets_interactors_items_recommended[dataset_preprocessor['name']][itr_class.__name__]
        for jj, metric_class_name in enumerate(metrics_classes_names):
            for zz, (ds_fieldname,ds_fieldvalue) in enumerate(ds_data.items()):
                # print(len(datasets_metrics_users_values[dataset_preprocessor['name']][
                    # metric_class_name][itr_class.__name__][-1]))
                metric_vals= datasets_metrics_users_values[dataset_preprocessor['name']][
                    metric_class_name][itr_class.__name__][-1]
                
                pearson = scipy.stats.pearsonr(datasets_metrics_users_values[dataset_preprocessor['name']][
                    metric_class_name][itr_class.__name__][-1],ds_fieldvalue[list(metric_vals.keys())])[0,0]
                results[2+(ii+1)*(zz+1)][(jj+1)*(hh+1)+2] = pearson
        # vals = []
        
        # results.append(['','G1','G2','G3',interactors_classes_names_to_names[itr_class_1.__name__],interactors_classes_names_to_names[itr_class_2.__name__]])
        # results.append([dataset_preprocessor['name'],hits_g1,hits_g2,hits_g3,hits_itr_1,hits_itr_2])
        # results.append(['Num. Items',len(g1),len(g2),len(g3),len(all_items),len(all_items)])
        # results.append(['Num. Mean Items',mni_g1,mni_g2,mni_g3,mni_itr_1,mni_itr_2])

# file_name=os.path.join(DirectoryDependent().DIRS['img'],f'dataset_analyses.png')
# lib.utils.utils.create_path_to_file(file_name)
# handles, labels = axs[0,0].get_legend_handles_labels()
# print(handles,labels)
# fig.legend(handles, labels, loc='upper center',ncol = 3,
          # fancybox=True, shadow=True)
# # fig.tight_layout()
# fig.savefig(file_name)
print(tabulate(results))

