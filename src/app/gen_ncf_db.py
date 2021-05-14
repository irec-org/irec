from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
import interactors
import random
import mf
from lib.utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from lib.utils.DatasetManager import DatasetManager
import yaml
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import time
from collections import defaultdict

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b',nargs='*')
args = parser.parse_args()

def main():
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
    if args.b == None:
        datasets_preprocessors = dm.request_datasets_preprocessors()
    else:
        datasets_preprocessors = [datasets_preprocessors[base] for base in args.b]
    for dataset_preprocessor in datasets_preprocessors:

        dm = DatasetManager()
        dm.initialize_engines(dataset_preprocessor)
        dm.load()
        data=  dm.dataset_preprocessed[0].data
        users_consumed_items = defaultdict(set)
        all_items = set(list(range(dm.dataset_preprocessed[0].num_items)))
        with open('{}.train.rating'.format(dataset_preprocessor['name']),'w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                iid = int(data[i,1])
                rating = data[i,2]
                timestamp = int(data[i,3])
                f.write('{}\t{}\t{}\t{}\n'.format(uid,iid,rating,timestamp))
                users_consumed_items[uid].add(iid)
        data=  dm.dataset_preprocessed[1].data
        with open('{}.test.rating'.format(dataset_preprocessor['name']),'w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                iid = int(data[i,1])
                rating = data[i,2]
                timestamp = int(data[i,3])
                f.write('{}\t{}\t{}\t{}\n'.format(uid,iid,rating,timestamp))
                users_consumed_items[uid].add(iid)
        
        users_negative_items = {uid: list(map(str,list(all_items-items))) for uid, items in users_consumed_items.items()}
            
        with open('{}.test.negative'.format(dataset_preprocessor['name']),'w') as f:
            for i in range(len(data)):
                uid = int(data[i,0])
                sampled_negative_items = random.sample(users_negative_items[uid],99)
                f.write('({},{})\t{}\n'.format(uid,iid,'\t'.join(sampled_negative_items)))


        # data=  dm.dataset_preprocessed[1].data
        # with open('test_ncf_{}.csv'.format(dataset_preprocessor['name']),'w+') as f:
            # for i in range(len(data)):
                # uid = int(data[i,0])
                # iid = int(data[i,1])
                # rating = data[i,2]
                # timestamp = int(data[i,3])
                # f.write('{}\t{}\t{}\t{}\n'.format(uid,iid,rating,timestamp))


if __name__ == '__main__':
    main()
