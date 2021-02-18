from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
import interactors
import mf
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from utils.DatasetManager import DatasetManager
from utils.DirectoryDependent import DirectoryDependent
import yaml
from metric import InteractionMetricsEvaluator, CumulativeMetricsEvaluator, CumulativeInteractionMetricsEvaluator, UserCumulativeInteractionMetricsEvaluator
from utils.dataset import Dataset
from utils.PersistentDataManager import PersistentDataManager
from utils.InteractorCache import InteractorCache
import metric
from utils.util import run_parallel
import ctypes
import matplotlib

font = {'size'   : 22}

matplotlib.rc('font', **font)

nums_interactions_to_print = [5, 10, 20]

dm = DatasetManager()
datasets_preprocessors = dm.request_datasets_preprocessors()
interactors_preprocessor_paramaters = yaml.load(
    open("settings" + sep + "interactors_preprocessor_parameters.yaml"),
    Loader=yaml.SafeLoader)
interactors_general_settings = yaml.load(
    open("settings" + sep + "interactors_general_settings.yaml"),
    Loader=yaml.SafeLoader)

evaluation_policies_parameters = yaml.load(
    open("settings" + sep + "evaluation_policies_parameters.yaml"),
    Loader=yaml.SafeLoader)

ir = InteractorRunner(dm, interactors_general_settings,
                      interactors_preprocessor_paramaters,
                      evaluation_policies_parameters)
interactors_classes = ir.select_interactors()
for dataset_preprocessor in datasets_preprocessors:
    
    dm.initialize_engines(dataset_preprocessor)
    dm.load()

    ir = InteractorRunner(dm, interactors_general_settings,
                          interactors_preprocessor_paramaters,
                          evaluation_policies_parameters)

    data = np.vstack(
        (dm.dataset_preprocessed[0].data, dm.dataset_preprocessed[1].data))

    dataset = Dataset(data)
    dataset.update_from_data()
    dataset.update_num_total_users_items()

    consumption_matrix = scipy.sparse.csr_matrix((dataset.data[:,2],(dataset.data[:,0],dataset.data[:,1])),(dataset.num_total_users,dataset.num_total_items))

    evaluation_policy = ir.get_interactors_evaluation_policy()
    items_popularity = interactors.MostPopular.get_items_popularity(consumption_matrix,normalize=False)
    items_entropy = interactors.Entropy.get_items_entropy(consumption_matrix)
    for itr_class in interactors_classes:
        fig, ax = plt.subplots()
        itr = ir.create_interactor(itr_class)
        pdm = PersistentDataManager(directory='results')
        results = pdm.load(InteractorCache().get_id(
            dm, evaluation_policy, itr))

        users_items_recommended = defaultdict(list)
        for uid, item in results:
            users_items_recommended[uid].append(item)
        position_most_recommended_item = dict()
        for pos in range(max(nums_interactions_to_print)):
            counter = defaultdict(int)
            for uid, items in users_items_recommended.items():
                counter[items[pos]] += 1

            top_item = max(counter, key=counter.get)
            position_most_recommended_item[pos] = top_item
        previous_pos = 0
        ax.scatter(items_entropy,
                items_popularity,
                color='gray')
        for pos in np.sort(nums_interactions_to_print):
            ax.scatter([items_entropy[position_most_recommended_item[i]] for i in range(previous_pos,pos)],
                    [items_popularity[position_most_recommended_item[i]] for i in range(previous_pos,pos)],
                    label=f'Top-{pos}',s=100)
            previous_pos = pos
        ax.set_title(interactors_general_settings[itr_class.__name__]['name'])
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Popularity")
        ax.set_xlim(xmin=np.min(items_entropy),xmax=np.max(items_entropy))
        ax.set_ylim(ymin=np.min(items_popularity),ymax=np.max(items_popularity))
        ax.legend()

        fig.savefig(os.path.join(DirectoryDependent().DIRS["img"],f'pop_ent_{dm.dataset_preprocessor.name}_{itr_class.__name__}.png'),bbox_inches = 'tight')

    # for metric_evaluator in metrics_evaluators:
        # args = [(id(metric_evaluator), id(dm), itr_class)
        # run_parallel(evaluate_itr, args, use_tqdm=False)
