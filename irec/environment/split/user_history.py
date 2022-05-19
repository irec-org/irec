import numpy as np
from .base import SplitStrategy
from typing import List, Tuple
from irec.environment.dataset import Dataset
from pandas import DataFrame
import pandas as pd


class UserHistory(SplitStrategy):
    """UserHistory.

    """

    def get_test_uids(
        self,
        data: np.ndarray,
        num_test_users: int
    ) -> np.array:
        """get_test_uids.

        Args:
            data (ndarray): the data to be split.
            num_test_users (int): total number of users in the test set.

        Returns:
            A list of the users IDs that will be in the test set.
        """
        data_df = pd.DataFrame(data, columns=["userId", "itemId", "rating", "timestamp"])
        test_uids = np.array(data_df["userId"].unique())
        return test_uids


    def split_dataset(self, data: np.ndarray, test_uids: List[int]) -> Tuple[Dataset, Dataset]:
        """split_dataset.

        Method responsible for performing the splitting.

        Args:
            data (np.ndarray): the data to be splitted.
            test_uids (List[int]): list of the users ID in the testset.
        Returns:
            A tuple containing the trainset and the testset.
        """

        def split_history(df: pd.DataFrame):
            ratings_train = int(len(df) * self.train_size)  
            train, test = df.iloc[:ratings_train], df.iloc[ratings_train:]
            testset.append(test)
            return train

        dataframe = DataFrame(data)
        
        testset = []
        trainset = dataframe.groupby(0).apply(
            lambda user_history: 
                split_history(user_history.sort_values(3)).reset_index(drop=True)
            
        ).reset_index(drop=True)

        testset = pd.concat(testset)

        train_dataset = Dataset(np.array(trainset))
        test_dataset = Dataset(np.array(testset))

        train_dataset.set_parameters()
        test_dataset.set_parameters()

        print("Test shape:", test_dataset.data.shape)
        print("Train shape:", train_dataset.data.shape)

        return train_dataset, test_dataset