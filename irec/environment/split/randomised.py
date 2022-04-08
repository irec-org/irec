import numpy as np
import pandas as pd

from .base import SplitStrategy
from random import sample
from pandas import DataFrame


class Random(SplitStrategy):
    """Random.

    A strategy that randomly selects data for the 
    training and testing sets;

    """

    def get_test_uids(
        self,
        data: np.ndarray,
        num_test_users: int
    ) -> np.array:
        """get_test_uids.
        Performs a random splitting.
        Args:
            data (ndarray): the data to be split.
            num_test_users (int): total number of users in the test set.
        Returns:
            A list of the users IDs that will be in the test set.
        """
        data_df = pd.DataFrame(data, columns=["userId", "itemId", "rating", "timestamp"])
        test_candidate_users = self._get_users_candidate(data_df)
        test_uids = np.array(sample(test_candidate_users,
                                    k=num_test_users))
        return test_uids
