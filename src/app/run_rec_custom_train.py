from os.path import dirname, realpath, sep, pardir
import os
import sys
from copy import copy
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")
import inquirer
import interactors
import mf
import util
from util import DatasetFormatter, MetricsEvaluator
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
import recommenders
import argparse
import evaluation_policy
parser = argparse.ArgumentParser()
parser.add_argument('--forced_run', default=False, action='store_true')
parser.add_argument('--parallel', default=False, action='store_true')
parser.add_argument('-m',nargs='*')
parser.add_argument('-b',nargs='*')
parser.add_argument('-efirst')
parser.add_argument('-elast')
args = parser.parse_args()

if args.e == None:
    first_evaluation_policy = ir.get_interactors_evaluation_policy()
else:
    first_evaluation_policy = eval('evaluation_policy.'+args.efirst)()

last_evaluation_policy = eval('evaluation_policy.'+args.elast)()
interactors_preprocessor_paramaters = yaml.load(
    open("settings" + sep + "interactors_preprocessor_parameters.yaml"),
    Loader=yaml.SafeLoader)
interactors_general_settings = yaml.load(
    open("settings" + sep + "interactors_general_settings.yaml"),
    Loader=yaml.SafeLoader)

evaluation_policies_parameters = yaml.load(
    open("settings" + sep + "evaluation_policies_parameters.yaml"),
    Loader=yaml.SafeLoader)

with open("settings"+sep+"datasets_preprocessors_parameters.yaml") as f:
    loader = yaml.SafeLoader
    datasets_preprocessors = yaml.load(f,Loader=loader)

    datasets_preprocessors = {setting['name']: setting
                              for setting in datasets_preprocessors}

dm = DatasetManager()
datasets_preprocessors = [datasets_preprocessors[base] for base in args.b]
ir = InteractorRunner(None, interactors_general_settings,
                      interactors_preprocessor_paramaters,
                      evaluation_policies_parameters)
interactors_classes = [eval('interactors.'+interactor) for interactor in args.m]
history_rates_to_train = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)
    dm.load()
    data = np.vstack(
        (dm.dataset_preprocessed[0].data, dm.dataset_preprocessed[1].data))
    dataset = Dataset(data)
    dataset.update_from_data()
    dataset.update_num_total_users_items()
    consumption_matrix = scipy.sparse.csr_matrix((dataset.data[:,2],(dataset.data[:,0].astype(int),dataset.data[:,1].astype(int))),shape=(dataset.num_total_users,dataset.num_total_items))
    for history_rate in history_rates_to_train:
        print('%.2f%% of history'%(history_rate*100))
        for interactor_class in interactors_classes:
            # ir = InteractorRunner(dm, interactors_general_settings,
                          # interactors_preprocessor_paramaters,
                          # evaluation_policies_parameters)
                        # interactions=100,
                        # interaction_size=5)
            # ir.run_interactors(interactors_classes,forced_run=args.forced_run, parallel=args.parallel)
            itr = interactor_class(**interactors_preprocessor_paramaters[datasets_preprocessors['name']][interactor_class.__name__])
            users_items_recommended = evaluation_policy.evaluate(
                itr, self.dm.dataset_preprocessed[0],
                self.dm.dataset_preprocessed[1])

            # interactor_model.results = interactor_model.load_results()
            # pdm = PersistentDataManager(directory='results')
            # users_items_recommended = pdm.load(InteractorCache().get_id(
                # dm, first_evaluation_policy, itr))
            new_data = []
            for (user, item) in users_items_recommended:
                new_data.append((user,item,consumption_matrix[user,item]))
            new_train_data = np.array(new_data)
            train_data = dm.dataset_preprocessed[0].data
            new_train_data = np.vstack(
                (train_data, new_data))
            new_test_data=test_data[~(np.isin(test_data[:,0],set(new_train_data[:,0])) & np.isin(test_data[:,1],set(new_train_data[:,1])))]
            train_dataset = copy(dataset)
            train_dataset.data = new_train_data
            train_dataset.update_from_data()
            test_dataset = copy(dataset)
            test_dataset.data = new_test_data
            test_dataset.update_from_data()

            pdm = PersistentDataManager(directory='results')
            # if forced_run or not pdm.file_exists(InteractorCache().get_id(
                    # self.dm, evaluation_policy, itr)):
            itr = interactor_class(**interactors_preprocessor_paramaters[datasets_preprocessors['name']][interactor_class.__name__])
            history_items_recommended = last_evaluation_policy.evaluate(
                itr, train_dataset,
                test_dataset)

            pdm = PersistentDataManager(directory='results')
            pdm.save(InteractorCache().get_id(self.dm, evaluation_policy, itr),
                     history_items_recommended)

            # users_items_recommended = evaluation_policy.evaluate(
                # itr, self.dm.dataset_preprocessed[0],
                # self.dm.dataset_preprocessed[1])

            # data = np.vstack(
                # (dm.dataset_preprocessed.data, dm.dataset_preprocessed[1].data))

            # print('\t*',interactor_class.__name__)
            # for recommender_class in recommenders_class:
                # print('\t\t-',recommender_class.__name__)
                # recommender_model = recommender_class(name_prefix=dsf.base,
                                                      # name_suffix=interactor_model.get_id()+'_history_rate_%.2f'%(history_rate))
                # recommender_model.train(train_matrix)
                # recommender_model.predict(test_matrix)
