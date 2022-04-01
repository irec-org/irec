import numpy as np

from .base import SplitStrategy
from pandas import DataFrame
from typing import List


class Temporal(SplitStrategy):
    """Random.

    A strategy that splits the user-item interactions based
    on timestamp. The idea is to avoid introducing a bias in users'
    consumption history by breaking the chronological order of the
    events.
    """

    def get_test_uids(
        self,
        data_df: DataFrame,
        num_test_users: int
    ) -> List[int]:
        """get_test_uids.

        Performs the temporal splitting strategy.

        Args:
            data_df (DataFrame): the data to be splitted.
            num_test_users (int): total number of users in the testset.
        Returns:
            A list of the users IDs that will be in the testset.
        """

        test_candidate_users = self._get_users_candidate(data_df)
        test_candidate_users = np.array(test_candidate_users, dtype=int)
        users_start_time = data_df.groupby(0).min()[3].to_numpy()
        test_uids = np.array(
            list(
                test_candidate_users[
                    list(reversed(np.argsort(users_start_time[test_candidate_users])))
                ]
            )[:num_test_users]
        )
        return test_uids
