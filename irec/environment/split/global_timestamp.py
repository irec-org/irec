import numpy as np

from .base import SplitStrategy

from typing import List, Tuple
from irec.environment.dataset import Dataset
from pandas import DataFrame


class GlobalTimestampSplit(SplitStrategy):
    """GlobalTimestampSplit.

    """

    def _get_sorted_first_interactions(self,
                                       df: DataFrame,
                                       ascending: bool = False
                                       ) -> DataFrame:
        """
        Returns a sorted dataframe that maps each user to the
        timestamp of their first interaction on the dataset.

        DataFrame format expected:
            - Column 0 = User ID
            - Column 1 = Item ID
            - Column 2 = Rating
            - Column 3 = Timestamp
        """
        mapping = df.groupby(0).agg({3: "min"})

        return mapping.sort_values(3, ascending=ascending).reset_index(0)

    def get_test_uids(self, data: np.ndarray, num_test_users: int) -> np.ndarray:
        """get_test_uids.

        Performs the global timestamp splitting strategy.

        Args:
            data (ndarray): the data to be split.
            num_test_users (int): total number of users in the test set.

        Returns:
            A list of the users IDs that will be in the test set.
        """
        dataframe = DataFrame(data)

        first_interactions = self._get_sorted_first_interactions(dataframe)
        sorted_users = np.array(first_interactions[0])

        total_users = len(sorted_users)

        num_train_users = round(total_users * self.train_size)
        num_test_users = int(total_users - num_train_users)

        return sorted_users[:num_test_users]

    def split_dataset(self, data: np.ndarray, test_uids: List[int]) -> Tuple[Dataset, Dataset]:
        """split_dataset.

        Method responsible for performing the splitting.

        Args:
            data (np.ndarray): the data to be splitted.
            test_uids (List[int]): list of the users ID in the testset.
        Returns:
            A tuple containing the trainset and the testset.
        """

        dataframe = DataFrame(data)

        first_interactions = self._get_sorted_first_interactions(dataframe)
        threshold = int(first_interactions.loc[first_interactions[0] == test_uids[-1]][3])

        trainset = dataframe[
            (dataframe[3] <= threshold) &
            (~dataframe[0].isin(test_uids))
        ]

        testset = dataframe[dataframe[0].isin(test_uids)]

        train_dataset = Dataset(np.array(trainset))
        test_dataset = Dataset(np.array(testset))

        train_dataset.set_parameters()
        test_dataset.set_parameters()

        print("Test shape:", test_dataset.data.shape)
        print("Train shape:", train_dataset.data.shape)

        return train_dataset, test_dataset