from copy import copy
from typing import Any
import numpy as np
import pandas as pd
import random


class Splitter:
    pass


class TrainTestConsumption(Splitter):
    def __init__(self, train_size=0.8, test_consumes=1, crono=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_size = train_size
        self.test_consumes = test_consumes
        self.crono = crono

    def process(self, dataset):
        data = dataset.data
        num_users = len(np.unique(data[:, 0]))
        num_train_users = round(num_users * (self.train_size))
        num_test_users = int(num_users - num_train_users)
        data_df = pd.DataFrame(data)
        users_items_consumed = data_df.groupby(0).count().iloc[:, 0]
        test_candidate_users: Any = list(
            users_items_consumed[users_items_consumed >= self.test_consumes]
            .to_dict()
            .keys()
        )
        if self.crono:
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
        else:
            test_uids = np.array(random.sample(test_candidate_users, k=num_test_users))
            # print(test_uids)
        # train_uids = np.array(list(set(range(num_users)) - set(test_uids)))

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


class TRTETrainValidation(Splitter):
    def __init__(self, train_size=0.8, test_consumes=1, crono=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_size = train_size
        self.test_consumes = test_consumes
        self.crono = crono

    def process(self, train_dataset, test_dataset):
        ttc = TrainTestConsumption(self.train_size, self.test_consumes, self.crono)
        train_dataset, test_dataset = ttc.process(train_dataset)
        return train_dataset, test_dataset
