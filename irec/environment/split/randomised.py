import numpy as np
import pandas as pd

from .base import SplitStrategy
from random import sample


class Random(SplitStrategy):

    def get_test_uids(self,
                      data_df: pd.DataFrame,
                      num_test_users: int):

        test_candidate_users = self._get_users_candidate(data_df)
        test_uids = np.array(sample(test_candidate_users,
                                    k=num_test_users))
        return test_uids
