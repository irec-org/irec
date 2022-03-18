import random
from copy import copy

import numpy as np
from utils import dataset as dataset_module


class SplitStrategy:
    def __init__(self, train_size, test_consumes):
        self.train_size = train_size
        self.test_consumes = test_consumes

    def _get_users_candidate(self, data_df):
        users_items_consumed = data_df.groupby(0).count().iloc[:, 0]
        test_candidate_users = list(
            users_items_consumed[users_items_consumed >= self.test_consumes]
            .to_dict()
            .keys()
        )
        return test_candidate_users

    def get_test_uids(self):
        pass

    def split_dataset(self, dataset, test_uids):

        data = dataset.data
        data[:, 0] = dataset_module.normalize_ids(data[:, 0])
        data[:, 1] = dataset_module.normalize_ids(data[:, 1])

        data_isin_test_uids = np.isin(data[:, 0], test_uids)

        train_dataset = copy(dataset)
        train_dataset.data = data[~data_isin_test_uids, :]
        dataset.set_parameters()

        test_dataset = copy(dataset)
        test_dataset.data = data[data_isin_test_uids, :]
        dataset.set_parameters()

        print("Test shape:", test_dataset.data.shape)
        print("Train shape:", train_dataset.data.shape)

        return dataset_module.TrainTestDataset(train=train_dataset, test=test_dataset)
