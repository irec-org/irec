import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-m', nargs='*')
parser.add_argument('-b', nargs='*')
settings = utils.load_settings()
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir)

import lib.evaluation_policies
import inquirer
import lib.value_functions
import lib.mf
from lib.utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from lib.utils.DatasetManager import DatasetManager
import yaml
from metrics import InteractionMetricsEvaluator, CumulativeMetricsEvaluator, CumulativeInteractionMetricsEvaluator, UserCumulativeInteractionMetricsEvaluator
from lib.utils.dataset import Dataset
from lib.utils.PersistentDataManager import PersistentDataManager
import metrics
from lib.utils.utils import run_parallel
import ctypes
import pandas as pd

dm = DatasetManager()

ir = InteractorRunner(dm, settings['interactors_general_settings'],
                      settings['agents_preprocessor_parameters'],
                      settings['evaluation_policies_parameters'])
# for dataset_preprocessor in datasets_preprocessors:
for base in args.b:
    dataset_preprocessor = settings['datasets_preprocessors_parameters'][base]
    dm.initialize_engines(dataset_preprocessor)
    dm.load()

    data = np.vstack(
        (dm.dataset_preprocessed[0].data, dm.dataset_preprocessed[1].data))

    dataset = Dataset(data)
    dataset.update_from_data()
    dataset.update_num_total_users_items()

    consumption_matrix = scipy.sparse.csr_matrix(
        (dataset.data[:, 2], (dataset.data[:, 0], dataset.data[:, 1])),
        (dataset.num_total_users, dataset.num_total_items))
    evaluation_policy_name = settings['defaults'][
        'interactors_evaluation_policy']
    evaluation_policy_parameters = settings['evaluation_policies_parameters'][
        evaluation_policy_name]
    evaluation_policy = eval('lib.evaluation_policies.' +
                             evaluation_policy_name)(
                                 **evaluation_policy_parameters)

    for agent_name in args.m:
        parameters = settings['agents_preprocessor_parameters'][base][
            agent_name]
        agent = utils.create_agent(agent_name, parameters)
        agent_id = utils.get_agent_id(agent_name, parameters)
        agent_methods= ','.join(list(list(parameters.values())[0]['agents'].keys()))
        pdm = PersistentDataManager(directory='results')
        users_items_recommended = pdm.load(
            utils.get_experiment_run_id(dm, evaluation_policy, agent_id))

        pdm = PersistentDataManager(directory='acts_info')
        acts_info = pdm.load(
            utils.get_experiment_run_id(dm, evaluation_policy, agent_id))
        if agent_name == 'NaiveEnsemble':
            data = []
            for i in range(len(acts_info)):
                uid = users_items_recommended[i][0]
                iid = users_items_recommended[i][1]
                data.append({
                    **acts_info[i],
                    **{
                        'uid': uid,
                        'iid': iid,
                        'reward': consumption_matrix[uid, iid]
                    }
                })

            df_results = pd.DataFrame(data)
            results = df_results.groupby(['user_interaction', 'meta_action_name'])['trial'].agg(['count'])
            print(df_results.head())
            print(results)

            results.to_csv(f'outputs/{base}_{agent_name}_{agent_methods}.csv')
            print(df_results.groupby('meta_action_name')['reward'].mean())
