from os.path import dirname, realpath, sep, pardir
import pickle
import os
import sys
import json
sys.path.append(dirname(realpath(__file__)) + sep + pardir)
import inquirer
from tabulate import tabulate
import copy
import pandas as pd
import seaborn as sn
import scipy
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

evaluation_policy_name = settings['defaults']['interactors_evaluation_policy']
evaluation_policy_parameters = settings['evaluation_policies_parameters'][evaluation_policy_name]
evaluation_policy=eval('lib.evaluation_policies.'+evaluation_policy_name)(**evaluation_policy_parameters)
datasets_interactors_items_recommended = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list)))

def get_groups_and_methods_metrics_from_sample(items,itr_recs,ground_truth_consumption_matrix):
    items=set(items)
    hits= 0
    for uid, rec_items in itr_recs.items():
        sample_rec_items = list(set(rec_items).intersection(items))
        hits+=np.sum(ground_truth_consumption_matrix[uid,sample_rec_items]>=4)
    hits/=len(itr_recs)
    return hits

for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)
    for itr_class in interactors_classes:
        itr = itr_class(**settings['interactors_preprocessor_paramaters'][dataset_preprocessor['name']][itr_class.__name__]['parameters'])
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
results = []
for dataset_preprocessor in datasets_preprocessors:
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
    for ii, itr_class_1 in enumerate(interactors_classes):
        # itr_1 = ir.create_interactor(itr_class_1)
        itr_1 = itr_class(**settings['interactors_preprocessor_paramaters'][dataset_preprocessor['name']][itr_class_1.__name__]['parameters'])
        for jj, itr_class_2 in enumerate(interactors_classes):
            if ii > jj:
                # itr_2 = ir.create_interactor(itr_class_1)
                itr_2 = itr_class(**settings['interactors_preprocessor_paramaters'][dataset_preprocessor['name']][itr_class_2.__name__]['parameters'])
                name = interactors_classes_names_to_names[itr_class_1.__name__]+r' $\times $ '+interactors_classes_names_to_names[itr_class_2.__name__]
                itr_1_recs = datasets_interactors_items_recommended[dataset_preprocessor['name']][itr_class_1.__name__]
                itr_2_recs = datasets_interactors_items_recommended[dataset_preprocessor['name']][itr_class_2.__name__]
                vals = []
                x = set()
                y = set()
                for uid1,items1 in itr_1_recs.items():
                    items2= itr_2_recs[uid1]
                    v = len(set(items1).intersection(set(items2)))/len(items1)
                    x |= set(itr_1_recs[uid1])
                    y |= set(itr_2_recs[uid1])
                    vals.append(v)
                
                vals=vals/np.max(vals)
                fig = utils.plot_similar_items(vals,'','',dataset_preprocessor['name'])
                file_name=os.path.join(DirectoryDependent().DIRS['img'],'similarity',f'{dataset_preprocessor["name"]}_{interactors_classes_names_to_names[itr_class_1.__name__]}_{interactors_classes_names_to_names[itr_class_2.__name__]}.png')
                lib.utils.utils.create_path_to_file(file_name)
                fig.savefig(file_name)
                g1 = x.intersection(y)
                g2 = x - y
                g3 = y - x
                all_items = list(range(ground_truth_dataset.num_total_items))
                hits_g1 = get_groups_and_methods_metrics_from_sample(g1,itr_1_recs,ground_truth_consumption_matrix)
                hits_g2 = get_groups_and_methods_metrics_from_sample(g2,itr_1_recs,ground_truth_consumption_matrix)
                hits_g3 = get_groups_and_methods_metrics_from_sample(g3,itr_1_recs,ground_truth_consumption_matrix)
                hits_itr_1 = get_groups_and_methods_metrics_from_sample(all_items,itr_1_recs,ground_truth_consumption_matrix)
                hits_itr_2 = get_groups_and_methods_metrics_from_sample(all_items,itr_2_recs,ground_truth_consumption_matrix)
                # results.append([dataset_preprocessor['name']])
                results.append(['','G1','G2','G3',interactors_classes_names_to_names[itr_class_1.__name__],interactors_classes_names_to_names[itr_class_2.__name__]])
                results.append([dataset_preprocessor['name'],hits_g1,hits_g2,hits_g3,hits_itr_1,hits_itr_2])

print(tabulate(results))
