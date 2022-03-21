import numpy as np

from typing import List
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

    @staticmethod
    def split_dataset(data: np.ndarray,
                      test_uids: List) -> [Dataset, Dataset]:

        data_isin_test_uids = np.isin(data[:, 0], test_uids)

        train_data = data[~data_isin_test_uids, :]
        train_dataset = Dataset(train_data)
        train_dataset.set_parameters()

        test_data = data[data_isin_test_uids, :]
        test_dataset = Dataset(test_data)
        test_dataset.set_parameters()

        print("Test shape:", test_dataset.data.shape)
        print("Train shape:", train_dataset.data.shape)

        return train_dataset, test_dataset
