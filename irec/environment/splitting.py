from copy import copy
import random
import numpy as np
import pandas as pd
from utils import dataset as dataset_module

class SplitStrategy():
    def __init__(self, train_size, test_consumes, strategy, *args, **kwargs):
        self.train_size = train_size
        self.test_consumes = test_consumes
        self.strategy = strategy

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
        data[:, 0] = dataset_module._si(data[:, 0])
        data[:, 1] = dataset_module._si(data[:, 1])

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



class Random(SplitStrategy):

    def get_test_uids(self, data_df, num_test_users):
        test_candidate_users = self._get_users_candidate(data_df)
        test_uids = np.array(random.sample(test_candidate_users, k=num_test_users))
        return test_uids

    def split_dataset(self, dataset, test_uids):
        return super().split_dataset(dataset, test_uids)

class Temporal(SplitStrategy):

    def get_test_uids(self, data_df, num_test_users):
        test_candidate_users = self._get_users_candidate(data_df)
        test_candidate_users = np.array(test_candidate_users, dtype=int)
        users_start_time = data_df.groupby(0).min()[3].to_numpy()
        test_uids = np.array(
            list(
                test_candidate_users[
                    list(
                        reversed(np.argsort(users_start_time[test_candidate_users]))
                    )
                ]
            )[:num_test_users]
        )
        return test_uids

    def split_dataset(self, dataset, test_uids):
        return super().split_dataset(dataset, test_uids)