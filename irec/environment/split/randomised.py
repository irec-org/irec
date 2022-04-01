import numpy as np

from .base import SplitStrategy
from random import sample
from typing import List
from pandas import DataFrame


class Random(SplitStrategy):
    """Random.

    A strategy that randomly selects data for the 
    training and testing sets;

    """

    def get_test_uids(
        self,
        data_df: DataFrame,
        num_test_users: int
    ) -> List[int]:
        """get_test_uids.

        Performs a random splitting.

        Args:
            data_df (DataFrame): the data to be splitted.
            num_test_users (int): total number of users in the testset.
        Returns:
            A list of the users IDs that will be in the testset.
        """

        test_candidate_users = self._get_users_candidate(data_df)
        test_uids = np.array(sample(test_candidate_users,
                                    k=num_test_users))
        return test_uids
