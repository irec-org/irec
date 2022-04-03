import numpy as np

from typing import List, Tuple
from irec.environment.dataset import Dataset
from pandas import DataFrame


class SplitStrategy:
    """SplitStrategy.

    Base class for the splitting strategies.
    """

    def __init__(self, train_size: float, test_consumes: int):
        """__init__.

        Args:
            train_size (float): defines the train size in percentage [0, 1]. 
            test_consumes (int): minimum number of items a user must consume to be a candidate.
        """

        self.train_size = train_size
        self.test_consumes = test_consumes

    def _get_users_candidate(self, data_df: DataFrame) -> List[int]:
        """_get_users_candidate.

        This method gets the IDs of the users with more than or equal
        to [self.test_consumes] consumptions.

        Args:
            data_df (DataFrame): the data to be splitted.
        Returns:
            List of the valid candidate users.
        """

        users_items_consumed = data_df.groupby("userId").count().iloc[:, 0]
        test_candidate_users = list(
            users_items_consumed[users_items_consumed >= self.test_consumes]
            .to_dict()
            .keys()
        )
        return test_candidate_users

    @staticmethod
    def split_dataset(
        data: np.ndarray,
        test_uids: List[int]
    ) -> Tuple[Dataset, Dataset]:
        """split_dataset.

        Method responsible for performing the splitting.

        Args:
            data (np.ndarray): the data to be splitted.
            test_uids (List[int]): list of the users ID in the testset.
        Returns:
            A tuple containing the trainset and the testset.
        """

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
