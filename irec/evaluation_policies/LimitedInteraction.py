from .EvaluationPolicy import EvaluationPolicy
from threadpoolctl import threadpool_limits
from collections import defaultdict
import scipy.sparse
import numpy as np
import random

class LimitedInteraction(EvaluationPolicy):
    def __init__(
        self, interaction_size, recommend_test_data_rate_limit, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.interaction_size = interaction_size
        self.recommend_test_data_rate_limit = recommend_test_data_rate_limit

    def evaluate(self, model, train_dataset, test_dataset):
        with threadpool_limits(limits=1, user_api="blas"):
            test_users = np.unique(test_dataset.data[:, 0]).astype(int)
            num_total_items = test_dataset.num_total_items
            test_consumption_matrix = scipy.sparse.csr_matrix(
                (
                    test_dataset.data[:, 2],
                    (
                        test_dataset.data[:, 0].astype(int),
                        test_dataset.data[:, 1].astype(int),
                    ),
                ),
                shape=(test_dataset.num_total_users, test_dataset.num_total_items),
            )

            users_items_recommended = defaultdict(list)
            num_test_users = len(test_users)
            print(f"Starting {model.name} Training")
            model.reset(train_dataset)
            print(f"Ended {model.name} Training")
            # users_num_interactions = defaultdict(int)
            users_num_items_to_recommend_from_test = dict()
            available_users = set()
            for uid in test_users:
                users_num_items_to_recommend_from_test[uid] = np.floor(
                    (test_consumption_matrix[uid] > 0).count_nonzero()
                    * self.recommend_test_data_rate_limit
                )
                if users_num_items_to_recommend_from_test[uid] > 0:
                    available_users |= {uid}

            users_num_items_recommended_from_test = defaultdict(int)

            history_items_recommended = []

            while len(available_users) > 0:
                uid = random.sample(available_users, k=1)[0]
                not_recommended = np.ones(num_total_items, dtype=bool)
                not_recommended[users_items_recommended[uid]] = 0
                items_not_recommended = np.nonzero(not_recommended)[0]
                # items_score, info = model.actions_estimate(
                # (uid, items_not_recommended))
                # best_items = items_not_recommended[np.argpartition(
                # items_score,
                # -self.interaction_size)[-self.interaction_size:]]
                actions, info = model.act(
                    (uid, items_not_recommended), self.interaction_size
                )
                best_items = actions[1]
                users_items_recommended[uid].extend(best_items)

                for item in best_items:
                    history_items_recommended.append((uid, item))
                    model.observe(
                        None, (uid, item), test_consumption_matrix[uid, item], info
                    )
                    users_num_items_recommended_from_test[uid] += (
                        test_consumption_matrix[uid, item] > 0
                    )

                # users_num_interactions[uid] += 1
                if (
                    users_num_items_recommended_from_test[uid]
                    >= users_num_items_to_recommend_from_test[uid]
                ):
                    available_users = available_users - {uid}

            return history_items_recommended, None