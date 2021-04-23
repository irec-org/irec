import os
from os.path import dirname, realpath, sep, pardir
import sys
from copy import copy
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--forced_run', default=False, action='store_true')
# parser.add_argument('--parallel', default=False, action='store_true')
parser.add_argument('-m',nargs='*')
parser.add_argument('-b',nargs='*')
parser.add_argument('--num_tasks', type=int, default=os.cpu_count())
parser.add_argument('-estart',default='LimitedInteraction')
parser.add_argument('-elast',default='Interaction')
parser.add_argument('-f1', default=False, action='store_true')
parser.add_argument('-f2', default=False, action='store_true')
args = parser.parse_args()
import inquirer
import interactors
import traceback
from utils.InteractorRunner import InteractorRunner
import joblib
import concurrent.futures
from utils.DatasetManager import DatasetManager
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import mf
import utils.util as util
# from util import DatasetFormatter, MetricsEvaluator
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
# import recommenders
import evaluation_policy
import yaml
import utils.dataset
from utils.InteractorCache import InteractorCache
from utils.PersistentDataManager import PersistentDataManager


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
# print(interactors_classes)
# history_rates_to_train = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
# history_rates_to_train = [0.1,0.3,0.5,0.6,0.8]
# history_rates_to_train = [0.1,0.3,0.5,0.8]
# history_rates_to_train = [0.1,0.3,0.5,0.6]
# history_rates_to_train = [0.1,0.3]
# history_rates_to_train = [0.8]
history_rates_to_train = [0.1]

def process(history_rate,dataset_preprocessor,dataset,consumption_matrix,dm,interactor_class):
    try:
        parameters = interactors_preprocessor_paramaters[dataset_preprocessor['name']][interactor_class.__name__]
        if parameters == None:
            parameters = dict()
        else:
            parameters = parameters['parameters']
        itr = interactor_class(**parameters)

        start_evaluation_policy = eval('evaluation_policy.'+args.estart)(**evaluation_policies_parameters[args.estart])
        start_evaluation_policy.recommend_test_data_rate_limit = history_rate
        # no need history rate s but i will put it because of consistency
        file_name_s = 's_'+str(history_rate)+'_'+InteractorCache().get_id(dm,start_evaluation_policy,itr)

        pdm = PersistentDataManager(directory='results')
        print(pdm.get_fp(file_name_s))
        if not pdm.file_exists(file_name_s) or args.f1:
            history_items_recommended = start_evaluation_policy.evaluate(
                itr, dm.dataset_preprocessed[0],
                dm.dataset_preprocessed[1])
            pdm.save(file_name_s,
                     history_items_recommended)

        parameters = interactors_preprocessor_paramaters[dataset_preprocessor['name']][interactor_class.__name__]
        if parameters == None:
            parameters = dict()
        else:
            parameters = parameters['parameters']
        itr = interactor_class(**parameters)

        last_evaluation_policy = eval('evaluation_policy.'+args.elast)(**evaluation_policies_parameters[args.elast])
        file_name = 'e_'+str(history_rate)+'_'+InteractorCache().get_id(dm,last_evaluation_policy,itr)

        print(pdm.get_fp(file_name))
        if not pdm.file_exists(file_name) or args.f2:
            if not(not pdm.file_exists(file_name_s) or args.f1):
                history_items_recommended = pdm.load(file_name_s)
                print("File already exists")
            new_data = []
            for (user, item) in history_items_recommended:
                if consumption_matrix[user,item]>0:
                    new_data.append((user,item,consumption_matrix[user,item]))
            new_train_data = np.array(new_data)
            train_data = dm.dataset_preprocessed[0].data
            new_train_data = np.vstack(
                    (train_data[:,0:3], new_data))
            test_data = dm.dataset_preprocessed[1].data
            new_test_data=test_data[~(np.isin(test_data[:,0],set(new_train_data[:,0])) & np.isin(test_data[:,1],set(new_train_data[:,1])))]
            train_dataset = copy(dataset)
            train_dataset.data = new_train_data
            train_dataset.update_from_data()
            test_dataset = copy(dataset)
            test_dataset.data = new_test_data
            test_dataset.update_from_data()
            history_items_recommended = last_evaluation_policy.evaluate(
                itr, train_dataset,
                test_dataset)
            pdm.save(file_name,
                     history_items_recommended)
        else:
            print("File already exists")
    except:
        print("Error Ocurred !!!!!!!!")
        traceback.print_exc()
        raise SystemError


with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_tasks) as executor:
    futures = set()
    for dataset_preprocessor in datasets_preprocessors:
        dm.initialize_engines(dataset_preprocessor)
        dm.load()
        data = np.vstack(
            (dm.dataset_preprocessed[0].data, dm.dataset_preprocessed[1].data))
        dataset = utils.dataset.Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        consumption_matrix = scipy.sparse.csr_matrix((dataset.data[:,2],(dataset.data[:,0].astype(int),dataset.data[:,1].astype(int))),shape=(dataset.num_total_users,dataset.num_total_items))
        for history_rate in history_rates_to_train:
            print('%.2f%% of history'%(history_rate*100))
            for interactor_class in interactors_classes:
                # process(history_rate,dataset_preprocessor,dataset,consumption_matrix,dm)
                    f=executor.submit(process,history_rate,dataset_preprocessor,dataset,consumption_matrix,dm,interactor_class)
                    futures.add(f)

                    if len(futures) >= args.num_tasks:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
    for f in futures:
        f.result()
