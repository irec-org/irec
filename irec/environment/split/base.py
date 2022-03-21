import numpy as np
import pandas as pd

from typing import List
from copy import copy
from irec.environment.dataset import Dataset


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

    def get_test_uids(self,
                      data_df: pd.DataFrame,
                      num_test_users: int):
        pass

    @staticmethod
    def split_dataset(dataset: Dataset,
                      test_uids: List):

        data = dataset.data
        data[:, 0] = dataset.normalize_ids(data[:, 0])
        data[:, 1] = dataset.normalize_ids(data[:, 1])

        data_isin_test_uids = np.isin(data[:, 0], test_uids)

        train_dataset = copy(dataset)
        train_dataset.data = data[~data_isin_test_uids, :]
        # TODO: @Thiago, I think here is train_dataset.set_parameters()
        dataset.set_parameters()

        test_dataset = copy(dataset)
        test_dataset.data = data[data_isin_test_uids, :]
        # TODO: @Thiago, I think here is test_dataset.set_parameters()
        dataset.set_parameters()

        print("Test shape:", test_dataset.data.shape)
        print("Train shape:", train_dataset.data.shape)

        return dataset_module.TrainTestDataset(train=train_dataset, test=test_dataset)
