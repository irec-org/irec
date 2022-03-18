from typing import List
from copy import copy
import pandas as pd
import scipy.sparse
import numpy as np
import random
import os

def normalize_ids(ids: List) -> np.array:
    unique_values = np.sort(np.unique(ids))
    result = np.searchsorted(unique_values, ids)
    return result

class Dataset:
    def __init__(
        self,
        data,
        num_total_users=None,
        num_total_items=None,
        num_users=None,
        num_items=None,
        rate_domain=None,
        uids=None,
    ):
        self.data = data
        self.num_users = num_users
        self.num_items = num_items
        self.rate_domain = rate_domain
        self.uids = uids
        self.num_total_users = num_total_users
        self.num_total_items = num_total_items

    def set_parameters(self):
        self.num_users = len(np.unique(self.data[:, 0]))
        self.num_items = len(np.unique(self.data[:, 1]))
        self.rate_domain = set(np.unique(self.data[:, 2]))
        self.uids = np.unique(self.data[:, 0]).astype(int)
        self.iids = np.unique(self.data[:, 1]).astype(int)
        self.max_uid = np.max(self.uids)
        self.max_iid = np.max(self.iids)
        self.mean_rating = np.mean(self.data[:, 2])
        self.min_rating = np.min(self.data[:, 2])
        self.max_rating = np.max(self.data[:, 2])
        self.num_total_users = self.max_uid + 1
        self.num_total_items = self.max_iid + 1

    def reset_index(self):
        self.data[:, 0] = normalize_ids(self.data[:, 0])
        self.data[:, 1] = normalize_ids(self.data[:, 1])

class TrainTestDataset:
    def __init__(self, train, test):
        self.train = train
        self.test = test


class TRTE():
    def process(self, dataset_dir):
        # dataset_dir = dataset_descriptor.dataset_dir
        train_data = np.loadtxt(os.path.join(dataset_dir, "train.data"), delimiter="::")
        test_data = np.loadtxt(os.path.join(dataset_dir, "test.data"), delimiter="::")

        dataset = Dataset(np.vstack([train_data, test_data]))
        dataset.set_parameters()
        
        train_dataset = copy(dataset)
        train_dataset.data = train_data
        train_dataset.set_parameters()
        test_dataset = copy(dataset)
        test_dataset.data = test_data
        test_dataset.set_parameters()
        return TrainTestDataset(train=train_dataset, test=test_dataset)


class DefaultDataset(Dataset):

    def __init__(self):
        pass

    def read(self, dataset_dir:str):
        data = np.loadtxt(
            os.path.join(dataset_dir, "ratings.csv"), delimiter=",", skiprows=1
        )
        dataset = Dataset(data)
        dataset.reset_index()
        dataset.set_parameters()
        return dataset

class MovieLensDataset(Dataset):

    def read(self, dataset_dir:str):
        data = np.loadtxt(
            os.path.join(dataset_dir, "ratings.csv"), delimiter="::", skiprows=1
        )
        data[:, 0] = normalize_ids(data[:, 0])
        data[:, 1] = normalize_ids(data[:, 1])
        dataset = Dataset(data)
        dataset.set_parameters()
        return dataset


"""
class TRTETrainValidation():
    def __init__(self, train_size, test_consumes, crono, random_seed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_size = train_size
        self.test_consumes = test_consumes
        self.crono = crono
        self.random_seed = random_seed

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        ttc = TrainTestConsumption(
            self.train_size, self.test_consumes, self.crono, self.random_seed
        )
        train_dataset, test_dataset = ttc.process(train_dataset)
        return train_dataset, test_dataset
"""