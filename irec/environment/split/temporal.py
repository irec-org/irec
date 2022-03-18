from .split_strategy import SplitStrategy
import numpy as np

class Temporal(SplitStrategy):

    def get_test_uids(self, data_df, num_test_users):
        test_candidate_users = self._get_users_candidate(data_df)
        test_candidate_users = np.array(test_candidate_users, dtype=int)
        users_start_time = data_df.groupby(0).min()[3].to_numpy()
        test_uids = np.array(
            list(
                test_candidate_users[
                    list(
                        reversed(np.argsort(users_start_time[test_candidate_users]))
                    )
                ]
            )[:num_test_users]
        )
        return test_uids

    def split_dataset(self, dataset, test_uids):
        return super().split_dataset(dataset, test_uids)