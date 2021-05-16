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
from sklearn.model_selection import GroupShuffleSplit
import interactors
import traceback
import random
from collections import defaultdict
from lib.utils.InteractorRunner import InteractorRunner
import joblib
import concurrent.futures
from lib.utils.DatasetManager import DatasetManager
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import mf
import lib.utils.utils as util
# from util import DatasetFormatter, MetricsEvaluator
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
# import recommenders
import evaluation_policies
import yaml
import lib.utils.dataset
from lib.utils.InteractorCache import InteractorCache
from lib.utils.PersistentDataManager import PersistentDataManager


interactors_preprocessor_parameters = yaml.load(
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
                      interactors_preprocessor_parameters,
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

def process(history_rate,dataset_preprocessor,dataset,consumption_matrix,dm,interactor_class,timestamp_matrix):
    try:

        data = dm.dataset_preprocessed[0].data
        with open('user_task_meta_train.csv','w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                iid = int(data[i,1])
                rating = data[i,2]
                f.write('{},{},{}\n'.format(uid,iid,rating))
        gss = GroupShuffleSplit(n_splits=1, train_size=.7, random_state=42)
        train_data= dm.dataset_preprocessed[0].data
        train_idx, test_idx = next(gss.split(train_data,groups=train_data[:,0].astype(int)))
        data = train_data[train_idx]
        with open('user_task_train_oracle_rating.csv','w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                iid = int(data[i,1])
                rating = data[i,2]
                f.write('{},{},{}\n'.format(uid,iid,rating))

        data = train_data[test_idx]
        with open('user_task_valid_oracle_rating.csv','w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                iid = int(data[i,1])
                rating = data[i,2]
                f.write('{},{},{}\n'.format(uid,iid,rating))


        data = dm.dataset_preprocessed[0].data
        with open('item_task_meta_train.csv','w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                iid = int(data[i,1])
                rating = data[i,2]
                f.write('{},{},{}\n'.format(uid,iid,rating))
        gss = GroupShuffleSplit(n_splits=1, train_size=.7, random_state=42)
        train_data= dm.dataset_preprocessed[0].data
        train_idx, test_idx = next(gss.split(train_data,groups=train_data[:,1].astype(int)))
        data = train_data[train_idx]
        with open('item_task_train_oracle_rating.csv','w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                iid = int(data[i,1])
                rating = data[i,2]
                f.write('{},{},{}\n'.format(uid,iid,rating))

        data = train_data[test_idx]
        with open('item_task_valid_oracle_rating.csv','w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                iid = int(data[i,1])
                rating = data[i,2]
                f.write('{},{},{}\n'.format(uid,iid,rating))

        parameters = interactors_preprocessor_parameters[dataset_preprocessor['name']][interactor_class.__name__]
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
        # if not pdm.file_exists(file_name_s) or args.f1:
            # history_items_recommended = start_evaluation_policy.evaluate(
                # itr, dm.dataset_preprocessed[0],
                # dm.dataset_preprocessed[1])
            # pdm.save(file_name_s,
                     # history_items_recommended)

        parameters = interactors_preprocessor_parameters[dataset_preprocessor['name']][interactor_class.__name__]
        if parameters == None:
            parameters = dict()
        else:
            parameters = parameters['parameters']
        itr = interactor_class(**parameters)

        last_evaluation_policy = eval('evaluation_policy.'+args.elast)(**evaluation_policies_parameters[args.elast])
        file_name = 'e_'+str(history_rate)+'_'+InteractorCache().get_id(dm,last_evaluation_policy,itr)

        if not(not pdm.file_exists(file_name_s) or args.f1):
            history_items_recommended = pdm.load(file_name_s)
            print("File already exists")

        data = np.array(history_items_recommended)
        with open('support.csv'.format(dataset_preprocessor['name']),'w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                iid = int(data[i,1])
                if consumption_matrix[uid,iid]>0:
                    rating = int(consumption_matrix[uid,iid]>0)
                    # timestamp = int(data[i,3])
                    f.write('{},{},{}\n'.format(uid,iid,rating))

        new_data = []
        for (user, item) in history_items_recommended:
            if consumption_matrix[user,item]>0:
                new_data.append((user,item,consumption_matrix[user,item],timestamp_matrix[user,item]))
        new_train_data = np.array(new_data)
        train_data = dm.dataset_preprocessed[0].data
        new_train_data = np.vstack(
                (train_data[:,0:4], new_data))
        test_data = dm.dataset_preprocessed[1].data
        new_test_data=test_data[~(np.isin(test_data[:,0],set(new_train_data[:,0])) & np.isin(test_data[:,1],set(new_train_data[:,1])))]
        train_dataset = copy(dataset)
        train_dataset.data = new_train_data
        train_dataset.update_from_data()
        test_dataset = copy(dataset)
        test_dataset.data = new_test_data
        test_dataset.update_from_data()

        all_items = set(list(range(train_dataset.num_items)))
        users_consumed_items = defaultdict(set)
        data= train_dataset.data
        print("Writing training data")

        with open('{}.train.rating'.format(dataset_preprocessor['name']),'w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                iid = int(data[i,1])
                rating = data[i,2]
                timestamp = int(data[i,3])
                f.write('{}\t{}\t{}\t{}\n'.format(uid,iid,rating,timestamp))
                users_consumed_items[uid].add(iid)
        data= test_dataset.data
            
        print("Test writing")
        with open('{}.test.rating'.format(dataset_preprocessor['name']),'w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                iid = int(data[i,1])
                rating = data[i,2]
                timestamp = int(data[i,3])
                f.write('{}\t{}\t{}\t{}\n'.format(uid,iid,rating,timestamp))
                users_consumed_items[uid].add(iid)
        print("Test writing finished")

        with open('query.csv'.format(dataset_preprocessor['name']),'w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                iid = int(data[i,1])
                if data[i,2]>0:
                    rating = int(data[i,2]>0)
                    # timestamp = int(data[i,3])
                    f.write('{},{},{}\n'.format(uid,iid,rating))
        
        # users_negative_items = {uid: list(map(str,list(all_items-items))) for uid, items in users_consumed_items.items()}
        # users_negative_items = {uid: list(map(str,list(all_items-items))) for uid, items in users_consumed_items.items()
            
        print("Test negative writing")
        with open('{}.test.negative'.format(dataset_preprocessor['name']),'w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                iid = int(data[i,1])
                # rating = data[i,2]
                # timestamp = int(data[i,3])
                user_negative_items = list(map(str,list(all_items-users_consumed_items[uid])))
                sampled_negative_items = random.sample(user_negative_items,99)
                f.write('({},{})\t{}\n'.format(uid,iid,'\t'.join(sampled_negative_items)))
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
        timestamp_matrix = scipy.sparse.csr_matrix((dataset.data[:,3],(dataset.data[:,0].astype(int),dataset.data[:,1].astype(int))),shape=(dataset.num_total_users,dataset.num_total_items))
        for history_rate in history_rates_to_train:
            print('%.2f%% of history'%(history_rate*100))
            for interactor_class in interactors_classes:
                # process(history_rate,dataset_preprocessor,dataset,consumption_matrix,dm)
                    f=executor.submit(process,history_rate,dataset_preprocessor,dataset,consumption_matrix,dm,interactor_class,timestamp_matrix)
                    futures.add(f)

                    if len(futures) >= args.num_tasks:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
    for f in futures:
        f.result()
