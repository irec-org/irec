import pandas as pd
import numpy as np
from collections import defaultdict
import random
import math
import time
import scipy.sparse
import os
import re
import numpy as np
import os
from copy import copy
from dataclasses import dataclass

class DatasetDescriptor:
    def __init__(self,name,dataset_dir):
        self.name = name
        self.dataset_dir = dataset_dir

class Dataset:
    def __init__(self,data,num_users=None,num_items=None,rate_domain=None,uids=None):
        self.data = data
        self.num_users = num_users
        self.num_items = num_items
        self.rate_domain = rate_domain
        self.uids = uids

    def update_from_data(self):
        self.num_users = len(np.unique(self.data[0]))
        self.num_items = len(np.unique(self.data[1]))
        self.rate_domain  = set(np.unique(self.data[2]))
        self.uids = np.unique(self.data[0])
        self.mean_rating = np.mean(self.data[2])

class DatasetParser:
    pass

class TRTE(DatasetParser):
    def parse_dataset(self,dataset_descriptor):
        dataset_dir = dataset_descriptor.dataset_dir
        train_data = np.loadtxt(os.path.join(dataset_dir,'train.data'),delimiter='::')
        test_data = np.loadtxt(os.path.join(dataset_dir,'test.data'),delimiter='::')

        dataset = Dataset(np.vstack([train_data,test_data]))
        dataset.update_from_data()
        train_dataset = copy(dataset)
        train_dataset.data = train_data
        test_dataset = copy(dataset)
        test_dataset.data = test_data
        return train_dataset, test_dataset

class MovieLens100k(DatasetParser):
    def parse_dataset(self,dataset_descriptor):
        dataset_dir = dataset_descriptor.dataset_dir
        data = np.loadtxt(os.path.join(dataset_dir,'u.data'),delimiter='\t')
        data[:,0] = data[:,0] - 1
        data[:,1] = data[:,1] - 1
        dataset = Dataset(data)
        dataset.update_from_data()
        return dataset

class MovieLens1M(DatasetParser):
    def parse_dataset(self,dataset_descriptor):
        dataset_dir = dataset_descriptor.dataset_dir
        data = np.loadtxt(os.path.join(dataset_dir,'ratings.dat'),delimiter='::')
        iids = dict()
        for i, iid in enumerate(df_cons[1].unique()):
            iids[iid] = i
        data[:,1] = np.vectorize(lambda x: iids[x])(data[1])
        data[:,0] = data[:,0] - 1
        dataset = Dataset(data)
        dataset.update_from_data()
        return dataset
