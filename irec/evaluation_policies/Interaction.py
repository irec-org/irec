from irec.CandidateActions import OneUserCandidateActions
from .EvaluationPolicy import EvaluationPolicy
from threadpoolctl import threadpool_limits
from irec.utils.dataset import Dataset
from collections import defaultdict
from irec.agents import Agent
from tqdm import tqdm
import scipy.sparse
import numpy as np
import random


class Interaction(EvaluationPolicy):
    def __init__(
        self,
        num_interactions: int,
        interaction_size: int,
        save_info: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_interactions = int(num_interactions)
        self.interaction_size = int(interaction_size)
        self.save_info = save_info

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

            for i in range(len(train_dataset.data)):
                uid = int(train_dataset.data[i, 0])
                if uid in test_users:
                    iid = int(train_dataset.data[i, 1])
                    reward = train_dataset.data[i, 2]
                    users_items_recommended[uid].append(iid)

            num_test_users = len(test_users)
            print(f"Starting {model.name} Training")
            model.reset(train_dataset)
            print(f"Ended {model.name} Training")
            users_num_interactions = defaultdict(int)
            available_users = set(test_users)

            history_items_recommended = []

            num_trials = num_test_users * self.num_interactions
            _intervals = num_trials // 200
            # _intervals = num_trials // num_trials
            _num_interactions = 0
            pbar = tqdm(total=num_trials)
            pbar.set_description(f"{model.name}")
            acts_info = []
            for i in range(num_trials):
                uid = random.sample(available_users, k=1)[0]
                not_recommended = np.ones(num_total_items, dtype=bool)
                not_recommended[users_items_recommended[uid]] = 0
                items_not_recommended = np.nonzero(not_recommended)[0]
                # items_score, info = model.action_estimates((uid,items_not_recommended))
                # best_items = items_not_recommended[np.argpartition(items_score,-self.interaction_size)[-self.interaction_size:]]

                actions, info = model.act(
                    OneUserCandidateActions(uid, items_not_recommended),
                    self.interaction_size,
                )
                if self.save_info:
                    info["trial"] = i
                    info["user_interaction"] = users_num_interactions[uid]
                acts_info.append(info)
                best_items = actions[1]
                users_items_recommended[uid].extend(best_items)

                for item in best_items:
                    history_items_recommended.append((uid, item))
                    model.observe(
                        None, (uid, item), test_consumption_matrix[uid, item], info
                    )
                # model.increment_time()
                users_num_interactions[uid] += 1
                if users_num_interactions[uid] == self.num_interactions:
                    available_users = available_users - {uid}

                _num_interactions += 1
                if  i != 0 and _intervals != 0 and i % _intervals == 0:
                    pbar.update(_num_interactions)
                    _num_interactions = 0
            pbar.update(_num_interactions)
            _num_interactions = 0
            pbar.close()
            return history_items_recommended, acts_info
