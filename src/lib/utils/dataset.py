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
from .Parameterizable import Parameterizable

class DatasetPreprocessor(Parameterizable):
    def __init__(self, name, dataset_descriptor,preprocessor,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.name = name
        self.dataset_descriptor = dataset_descriptor
        self.preprocessor = preprocessor
        self.parameters.extend(['dataset_descriptor','preprocessor'])

    def get_id(self,*args,**kwargs):
        return super().get_id(len(self.parameters),*args,**kwargs)

class Preprocessor(Parameterizable):
    def __init__(self,dataset_parser,splitter,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # self.dataset_descriptor = dataset_descriptor
        self.dataset_parser = dataset_parser
        self.splitter = splitter
        self.parameters.extend(['splitter','dataset_parser'])
    
class DatasetDescriptor(Parameterizable):
    def __init__(self,dataset_dir,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dataset_dir = dataset_dir
        self.parameters.extend(['dataset_dir'])

class Dataset:
    def __init__(self,data,num_users=None,num_items=None,rate_domain=None,uids=None):
        self.data = data
        self.num_users = num_users
        self.num_items = num_items
        self.rate_domain = rate_domain
        self.uids = uids

    def update_from_data(self):
        self.num_users = len(np.unique(self.data[:,0]))
        self.num_items = len(np.unique(self.data[:,1]))
        self.rate_domain  = set(np.unique(self.data[:,2]))
        self.uids = np.unique(self.data[:,0])
        self.mean_rating = np.mean(self.data[:,2])
        # self.consumption_matrix = scipy.sparse.csr_matrix((self.data[:,2],(self..data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.users_num,self.train_dataset.items_num))

class DatasetParser(Parameterizable):
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


def _netflix_read_ratings(self, fileName):
    file = open(fileName, "r")
    file.readline()
    numratings = np.sum([1 for line in open(fileName)])
    usersId = np.zeros(numratings, dtype=np.int32)
    itemsId = np.zeros(numratings, dtype=np.int32)
    ratings = np.zeros(numratings, dtype=np.float16)
    timestamp = np.zeros(numratings, dtype=np.int32)

    file = open(fileName, "r")
    file.readline()
    cont = 0
    for row in file:
        values = row.split('::')
        uid, iid,rating, ts = int(float(values[0])),int(float(values[1])),values[2], int(float(values[3].replace('\n', '')))
        usersId[cont] = uid
        itemsId[cont] = iid
        ratings[cont] = rating
        timestamp[cont] = ts
        cont += 1

    print(numratings,usersId[-1],itemsId[-1],ratings[-1])
    file.close()
    return usersId, itemsId, ratings, timestamp, numratings

class Netflix:
    def parse_dataset(self,dataset_descriptor):
        # base_dir = self.BASES_DIRS[self.base]
        u_train, i_train, r_train, t_train, numr_train = _netflix_read_ratings(dataset_descriptor.dataset_dir+'train.data')
        u_test, i_test, r_test, t_test, numr_test = _netflix_read_ratings(dataset_descriptor.dataset_dir+'test.data')
        test_data = np.array((u_test,i_test,r_test,t_test))
        train_data = np.array((u_train,i_train,r_train,t_train))
        return train_data, test_data
