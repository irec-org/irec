import numpy as np
import pandas as pd

from .base import SplitStrategy
from pandas import DataFrame


class Temporal(SplitStrategy):
    """Random.

    A strategy that splits the user-item interactions based
    on timestamp. The idea is to avoid introducing a bias in users'
    consumption history by breaking the chronological order of the
    events.
    """

    def get_test_uids(
        self,
        data: np.ndarray,
        num_test_users: int
    ) -> np.array:
        """get_test_uids.
        Performs the temporal splitting strategy.
        Args:
            data (ndarray): the data to be split.
            num_test_users (int): total number of users in the test set.
        Returns:
            A list of the users IDs that will be in the test set.
        """
        data_df = pd.DataFrame(data, columns=["userId", "itemId", "rating", "timestamp"])
        test_candidate_users = self._get_users_candidate(data_df)
        test_candidate_users = np.array(test_candidate_users, dtype=int)
        users_start_time = data_df.groupby("userId").min()["timestamp"].to_numpy()
        test_uids = np.array(
            list(
                test_candidate_users[
                    list(reversed(np.argsort(users_start_time[test_candidate_users])))
                ]
            )[:num_test_users]
        )
        return test_uids
