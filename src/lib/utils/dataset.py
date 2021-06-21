import pandas as pd
import numpy as np
import lib.value_functions
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
from .Parameterizable import Parameterizable


class DatasetPreprocessor(Parameterizable):
    def __init__(self, name, dataset_descriptor, preprocessor, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.dataset_descriptor = dataset_descriptor
        self.preprocessor = preprocessor
        self.parameters.extend(['dataset_descriptor', 'preprocessor'])

    def get_id(self, *args, **kwargs):
        return super().get_id(len(self.parameters), *args, **kwargs)


class Pipeline(Parameterizable):
    def __init__(self, steps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if steps is None:
            self.steps = []
        else:
            self.steps = steps
        self.parameters.extend(['steps'])

    def process(self, data):
        buf = data
        for element in self.steps:
            buf = element.process(buf)
        return buf


class DatasetDescriptor(Parameterizable):
    def __init__(self, dataset_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_dir = dataset_dir
        self.parameters.extend(['dataset_dir'])


class Dataset:
    def __init__(self,
                 data,
                 num_total_users=None,
                 num_total_items=None,
                 num_users=None,
                 num_items=None,
                 rate_domain=None,
                 uids=None):
        self.data = data
        self.num_users = num_users
        self.num_items = num_items
        self.rate_domain = rate_domain
        self.uids = uids
        self.num_total_users = num_total_users
        self.num_total_items = num_total_items

    def update_from_data(self):
        self.num_users = len(np.unique(self.data[:, 0]))
        self.num_items = len(np.unique(self.data[:, 1]))
        self.rate_domain = set(np.unique(self.data[:, 2]))
        self.uids = np.unique(self.data[:, 0])
        self.mean_rating = np.mean(self.data[:, 2])
        self.min_rating = np.min(self.data[:, 2])
        self.max_rating = np.max(self.data[:, 2])

    def update_num_total_users_items(self):
        self.num_total_users = self.num_users
        self.num_total_items = self.num_items

        # self.consumption_matrix = scipy.sparse.csr_matrix((self.data[:,2],(self..data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.users_num,self.train_dataset.items_num))


class DataProcessor(Parameterizable):
    pass


class TRTE(DataProcessor):
    def process(self, dataset_descriptor):
        dataset_dir = dataset_descriptor.dataset_dir
        train_data = np.loadtxt(os.path.join(dataset_dir, 'train.data'),
                                delimiter='::')
        test_data = np.loadtxt(os.path.join(dataset_dir, 'test.data'),
                               delimiter='::')

        dataset = Dataset(np.vstack([train_data, test_data]))
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        train_dataset = copy(dataset)
        train_dataset.data = train_data
        train_dataset.update_from_data()
        test_dataset = copy(dataset)
        test_dataset.data = test_data
        test_dataset.update_from_data()
        return train_dataset, test_dataset


class TRTEPopular(DataProcessor):
    def __init__(self, items_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items_rate = items_rate
        self.parameters.extend(['items_rate'])

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        data = np.vstack((test_dataset.data, train_dataset.data))
        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        num_items_to_sample = int(self.items_rate * dataset.num_total_items)
        consumption_matrix = scipy.sparse.csr_matrix(
            (dataset.data[:, 2], (dataset.data[:, 0], dataset.data[:, 1])),
            (dataset.num_total_users, dataset.num_total_items))
        items_popularity = value_functions.MostPopular.get_items_popularity(
            consumption_matrix)
        top_popular_items = np.argsort(
            items_popularity)[::-1][num_items_to_sample]
        test_dataset.data = test_dataset.data[test_dataset.data[:, 1].isin(
            top_popular_items)]
        test_dataset.update_from_data()
        train_dataset.data = train_dataset.data[train_dataset.data[:, 1].isin(
            top_popular_items)]
        train_dataset.update_from_data()

        # train_dataset.data[train_dataset.data[:,1].isin(top_popular_items)]

        train_dataset, test_dataset = ttc.process(train_dataset)
        return train_dataset, test_dataset


class TRTERandom(DataProcessor):
    def __init__(self, min_ratings, random_seed, probability_keep_item, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.min_ratings = min_ratings
        self.random_seed = random_seed
        self.probability_keep_item = probability_keep_item
        self.parameters.extend(
            ['min_ratings', 'random_seed', 'probability_keep_item'])

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        # ttc = TrainTestConsumption(self.train_size, self.test_consumes,
        # self.crono, self.random_seed)
        train_dataset, test_dataset = ttc.process(train_dataset)
        return train_dataset, test_dataset


class MovieLens100k(DataProcessor):
    def process(self, dataset_descriptor):
        dataset_dir = dataset_descriptor.dataset_dir
        data = np.loadtxt(os.path.join(dataset_dir, 'u.data'), delimiter='\t')
        data[:, 0] = data[:, 0] - 1
        data[:, 1] = data[:, 1] - 1
        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        return dataset


class MovieLens1M(DataProcessor):
    def process(self, dataset_descriptor):
        dataset_dir = dataset_descriptor.dataset_dir
        data = np.loadtxt(os.path.join(dataset_dir, 'ratings.dat'),
                          delimiter='::')
        iids = dict()
        for i, iid in enumerate(np.unique(data[:, 1])):
            iids[iid] = i
        data[:, 1] = np.vectorize(lambda x: iids[x])(data[:, 1])
        data[:, 0] = data[:, 0] - 1
        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
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
        uid, iid, rating, ts = int(float(values[0])), int(float(
            values[1])), values[2], int(float(values[3].replace('\n', '')))
        usersId[cont] = uid
        itemsId[cont] = iid
        ratings[cont] = rating
        timestamp[cont] = ts
        cont += 1

    print(numratings, usersId[-1], itemsId[-1], ratings[-1])
    file.close()
    return usersId, itemsId, ratings, timestamp, numratings


class Netflix:
    def process(self, dataset_descriptor):
        # base_dir = self.BASES_DIRS[self.base]
        u_train, i_train, r_train, t_train, numr_train = _netflix_read_ratings(
            dataset_descriptor.dataset_dir + 'train.data')
        u_test, i_test, r_test, t_test, numr_test = _netflix_read_ratings(
            dataset_descriptor.dataset_dir + 'test.data')
        test_data = np.array((u_test, i_test, r_test, t_test))
        train_data = np.array((u_train, i_train, r_train, t_train))
        return train_data, test_data


class TrainTestConsumption(DataProcessor):
    def __init__(self, train_size, test_consumes, crono, random_seed, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.train_size = train_size
        self.test_consumes = test_consumes
        self.crono = crono
        self.random_seed = random_seed
        self.parameters.extend(
            ['train_size', 'test_consumes', 'crono', 'random_seed'])

    def process(self, dataset):
        np.random.seed(self.random_seed)
        data = dataset.data
        num_users = len(np.unique(data[:, 0]))
        num_train_users = round(num_users * (self.train_size))
        num_test_users = int(num_users - num_train_users)
        data_df = pd.DataFrame(data)
        users_items_consumed = data_df.groupby(0).count().iloc[:, 0]
        test_candidate_users = list(users_items_consumed[
            users_items_consumed >= self.test_consumes].to_dict().keys())
        if self.crono:
            users_start_time = data_df.groupby(0).min()[3].to_numpy()
            test_uids = np.array(
                list(test_candidate_users[list(
                    reversed(np.argsort(
                        users_start_time[test_candidate_users])))])
                [:num_test_users])
        else:
            test_uids = np.array(
                random.sample(test_candidate_users, k=num_test_users))
            # print(test_uids)
        train_uids = np.array(list(set(range(num_users)) - set(test_uids)))

        data_isin_test_uids = np.isin(data[:, 0], test_uids)

        train_dataset = copy(dataset)
        train_dataset.data = data[~data_isin_test_uids, :]
        dataset.update_from_data()
        test_dataset = copy(dataset)
        test_dataset.data = data[data_isin_test_uids, :]
        dataset.update_from_data()
        print("Test shape:", test_dataset.data.shape)
        print("Train shape:", train_dataset.data.shape)
        return train_dataset, test_dataset


class TRTETrainValidation(DataProcessor):
    def __init__(self, train_size, test_consumes, crono, random_seed, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.train_size = train_size
        self.test_consumes = test_consumes
        self.crono = crono
        self.random_seed = random_seed
        self.parameters.extend(
            ['train_size', 'test_consumes', 'crono', 'random_seed'])

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        ttc = TrainTestConsumption(self.train_size, self.test_consumes,
                                   self.crono, self.random_seed)
        train_dataset, test_dataset = ttc.process(train_dataset)
        return train_dataset, test_dataset


class TRTESample(DataProcessor):
    def __init__(self, items_rate, sample_method, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items_rate = items_rate
        self.sample_method = sample_method
        self.parameters.extend(['items_rate', 'sample_method'])

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        dataset = Dataset(np.vstack([train_dataset.data, test_dataset.data]))
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        consumption_matrix = scipy.sparse.csr_matrix(
            (dataset.data[:, 2], (dataset.data[:, 0], dataset.data[:, 1])),
            (dataset.num_total_users, dataset.num_total_items))
        num_items_to_sample = int(self.items_rate * dataset.num_total_items)
        if self.sample_method == 'entropy':
            items_values = value_functions.Entropy.get_items_entropy(
                consumption_matrix)
        elif self.sample_method == 'popularity':
            items_values = value_functions.MostPopular.get_items_popularity(
                consumption_matrix)

        best_items = np.argpartition(
            items_values, -num_items_to_sample)[-num_items_to_sample:]
        dataset.data = dataset.data[np.isin(dataset.data[:, 1], best_items), :]

        new_iids = dict()
        for i, iid in enumerate(np.unique(dataset.data[:, 1])):
            new_iids[iid] = i
        for i in range(len(dataset.data)):
            dataset.data[i, 1] = new_iids[dataset.data[i, 1]]

        dataset.update_from_data()
        dataset.update_num_total_users_items()

        train_uids = train_dataset.uids
        test_uids = test_dataset.uids

        train_dataset = copy(dataset)
        train_dataset.data = dataset.data[np.isin(dataset.data[:, 0],
                                                  train_uids)]
        train_dataset.update_from_data()

        test_dataset = copy(dataset)
        test_dataset.data = dataset.data[np.isin(dataset.data[:, 0],
                                                 test_uids)]
        test_dataset.update_from_data()
        return train_dataset, test_dataset


class PopularityFilter(DataProcessor):
    def __init__(self, keep_popular, num_items_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_popular = keep_popular
        self.num_items_threshold = num_items_threshold
        self.parameters.extend(['keep_popular', 'num_items_threshold'])

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        dataset = Dataset(np.vstack([train_dataset.data, test_dataset.data]))
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        consumption_matrix = scipy.sparse.csr_matrix(
            (dataset.data[:, 2], (dataset.data[:, 0], dataset.data[:, 1])),
            (dataset.num_total_users, dataset.num_total_items))
        # num_items_to_sample = int(self.items_rate * dataset.num_total_items)
        items_values = lib.value_functions.MostPopular.get_items_popularity(
            consumption_matrix)
        items_sorted = np.argsort(items_values)[::-1]
        if self.keep_popular:
            items_to_keep = items_sorted[:self.num_items_threshold]
        else:
            items_to_keep = items_sorted[self.num_items_threshold:]

        dataset.data = dataset.data[np.isin(dataset.data[:,
                                                         1], items_to_keep), :]

        new_iids = dict()
        for i, iid in enumerate(np.unique(dataset.data[:, 1])):
            new_iids[iid] = i
        for i in range(len(dataset.data)):
            dataset.data[i, 1] = new_iids[dataset.data[i, 1]]
        new_uids = dict()
        for i, uid in enumerate(np.unique(dataset.data[:, 0])):
            new_uids[uid] = i
        for i in range(len(dataset.data)):
            dataset.data[i, 0] = new_uids[dataset.data[i, 0]]

        dataset.update_from_data()
        dataset.update_num_total_users_items()

        # train_uids = train_dataset.uids
        # test_uids = test_dataset.uids

        # train_dataset = copy(dataset)
        # train_dataset.data = dataset.data[np.isin(dataset.data[:, 0],
        # train_uids)]
        # train_dataset.update_from_data()

        # test_dataset = copy(dataset)
        # test_dataset.data = dataset.data[np.isin(dataset.data[:, 0], test_uids)]
        # test_dataset.update_from_data()
        return dataset


class CombineTrainTest(DataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        dataset = Dataset(np.vstack([train_dataset.data, test_dataset.data]))
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        return dataset
class PopRemoveEnt(DataProcessor):
    def __init__(self, num_items_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.keep_popular = keep_popular
        self.num_items_threshold = num_items_threshold
        self.parameters.extend(['num_items_threshold'])

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        dataset = Dataset(np.vstack([train_dataset.data, test_dataset.data]))
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        consumption_matrix = scipy.sparse.csr_matrix(
            (dataset.data[:, 2], (dataset.data[:, 0], dataset.data[:, 1])),
            (dataset.num_total_users, dataset.num_total_items))
        # num_items_to_sample = int(self.items_rate * dataset.num_total_items)
        items_values = lib.value_functions.MostPopular.get_items_popularity(
            consumption_matrix)
        items_sorted = np.argsort(items_values)[::-1]
        # if self.keep_popular:
        items_to_keep = items_sorted[:self.num_items_threshold]
        # else:
            # items_to_keep = items_sorted[self.num_items_threshold:]
        train_dataset.data[train_dataset.data[:,1].isin(items_to_keep),2] = 5
        test_dataset.data[test_dataset.data[:,1].isin(items_to_keep),2] = 5
        # dataset.update_from_data()
        # dataset.update_num_total_users_items()
        return dataset
