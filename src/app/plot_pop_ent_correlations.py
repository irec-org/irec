from os.path import dirname, realpath, sep, pardir
import scipy.stats
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
    
    items_entropy = lib.interactors.Entropy.get_items_entropy(ground_truth_consumption_matrix)
    items_popularity = lib.interactors.MostPopular.get_items_popularity(ground_truth_consumption_matrix,normalize=False)
    print(dataset_preprocessor['name'])
    print("\tpopularity correlation with entropy = %s"% str(scipy.stats.pearsonr(items_popularity,items_entropy)))
    items_logpopent = lib.interactors.LogPopEnt.get_items_logpopent(items_popularity,items_entropy)
    pe = scipy.stats.pearsonr(items_logpopent,items_entropy)
    pp = scipy.stats.pearsonr(items_logpopent,items_popularity)
    print("\tent*log(pop) correlation with entropy = %s"% (str(pe)))
    print("\tent*log(pop) correlation with popularity = %s"% (str(pp)))
    for k in np.linspace(0,1,11):
        print('----',k,'----')
        items_logpopent = lib.interactors.LogPopEnt.get_items_logpopent(items_popularity,items_entropy,k=k)
        pe = scipy.stats.pearsonr(items_logpopent,items_entropy)
        pp = scipy.stats.pearsonr(items_logpopent,items_popularity)
        print("\tent^%.1f*log(pop)^%.1f correlation with entropy = %s"% (k,1-k,str(pe)))
        print("\tent^%.1f*log(pop)^%.1f correlation with popularity = %s"% (k,1-k,str(pp)))
