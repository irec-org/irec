from .split_strategy import SplitStrategy
import random
import numpy as np

class Random(SplitStrategy):

    def get_test_uids(self, data_df, num_test_users):
        test_candidate_users = self._get_users_candidate(data_df)
        test_uids = np.array(random.sample(test_candidate_users, k=num_test_users))
        return test_uids

    def split_dataset(self, dataset, test_uids):
        return super().split_dataset(dataset, test_uids)
