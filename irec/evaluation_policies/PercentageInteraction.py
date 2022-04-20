from irec.agents.action import OneUserItemCollection
from irec.agents.value_functions.most_popular import MostPopular
from irec.agents.value_functions.log_pop_ent import LogPopEnt
from irec.agents.value_functions.best_rated import BestRated
from irec.agents.value_functions.entropy import Entropy
from concurrent.futures import ProcessPoolExecutor
from .EvaluationPolicy import EvaluationPolicy
from irec.environment.dataset import Dataset
from collections import defaultdict
from tqdm import tqdm
import scipy.sparse
import numpy as np
import itertools
import ctypes
import random
import copy
import time


class PercentageInteraction(EvaluationPolicy):
    def __init__(
        self,
        num_interactions: int,
        interaction_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_interactions = int(num_interactions)
        self.interaction_size = int(interaction_size)

    @staticmethod
    def rec_npers(parameters):
        obj_id, method, exchange_point, uid = parameters
        self = ctypes.cast(obj_id, ctypes.py_object).value
        max_items = int(len(self.test_consumption_matrix[uid, :].data) * exchange_point)
        not_recommended = np.ones(self.num_total_items, dtype=bool)
        user_items_recommended = self.users_items_recommended_original[uid].copy()
        not_recommended[user_items_recommended] = 0
        items_not_recommended = np.nonzero(not_recommended)[0]

        num_items = 0
        rec_items = []
        top_k_items = (
            random.sample(list(items_not_recommended), len(list(items_not_recommended)))
            if method == "Random"
            else self.nonp_methods[method]
        )

        for itr, item in enumerate(top_k_items):

            if self.test_consumption_matrix[uid, item] >= self.threshold:
                not_recommended[item] = 0
                num_items += 1
            rec_items.append([uid, item, self.test_consumption_matrix[uid, item], -1])
            if num_items == max_items:
                break

        items_not_recommended = np.nonzero(not_recommended)[0]
        return [
            uid,
            int((itr / self.interaction_size_npers) + 0.5),
            rec_items,
            items_not_recommended,
            not_recommended,
        ]

    def rec_mab(self, result, model):

        uid, num_itr_npers, rec_items, items_not_recommended, not_recommended = result
        user_items_recommended = []
        for itr in range(self.num_interactions):
            actions, info = model.act(OneUserItemCollection(uid, items_not_recommended), self.interaction_size)
            best_items = actions[1]
            for item in best_items:
                not_recommended[item] = 0
                user_items_recommended.append(item)
                if self.test_consumption_matrix[uid, item] >= self.threshold:
                    model.observe(
                        None, (uid, item), self.test_consumption_matrix[uid, item], info
                    )

            items_not_recommended = np.nonzero(not_recommended)[0]
        return [uid, num_itr_npers, user_items_recommended]

    def evaluate(self, model, train_dataset, test_dataset):
        start_time = time.time()

        self.dataset = "MovieLens 100k"
        self.threshold = 1
        self.interaction_size_npers = 5

        print("\ndataset", self.dataset)
        print("threshold", self.threshold)
        print("num_interactions", self.num_interactions)
        print("interaction_size", self.interaction_size)

        test_users = np.unique(test_dataset.data[:, 0]).astype(int)
        self.num_total_items = test_dataset.num_total_items
        self.test_consumption_matrix = scipy.sparse.csr_matrix(
            (
                test_dataset.data[:, 2],
                (
                    test_dataset.data[:, 0].astype(int),
                    test_dataset.data[:, 1].astype(int),
                ),
            ),
            shape=(test_dataset.num_total_users, test_dataset.num_total_items),
        )

        self.users_items_recommended_original = defaultdict(list)

        for i in range(len(train_dataset.data)):
            uid = int(train_dataset.data[i, 0])
            if uid in test_users:
                iid = int(train_dataset.data[i, 1])
                reward = train_dataset.data[i, 2]
                self.users_items_recommended_original[uid].append(iid)

        num_test_users = len(test_users)
        available_users = set(test_users)
        train_consumption_matrix = scipy.sparse.csr_matrix(
            (
                train_dataset.data[:, 2],
                (train_dataset.data[:, 0], train_dataset.data[:, 1]),
            ),
            (train_dataset.num_total_users, train_dataset.num_total_items),
        )
        items_entropy = Entropy.get_items_entropy(train_consumption_matrix)
        items_popularity = MostPopular.get_items_popularity(
            train_consumption_matrix, normalize=False
        )
        items_logPopEnt = LogPopEnt.get_items_logpopent(items_popularity, items_entropy)
        items_bestRated = BestRated.get_items_bestrated(train_consumption_matrix)

        self.exchange_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.nonp_methods = {
            "BestRated": np.argsort(items_bestRated)[::-1].astype(np.int32),
            "Entropy": np.argsort(items_entropy)[::-1].astype(np.int32),
            "Popularity": np.argsort(items_popularity)[::-1].astype(np.int32),
            "LogPopEnt": np.argsort(items_logPopEnt)[::-1].astype(np.int32),
            "Random": None,
        }

        self.uid_teste = list(available_users)[0]
        print(self.uid_teste)
        history_items_recommended = {}
        # available_users = list(available_users)[:10]
        itr = len(self.nonp_methods) * len(self.exchange_points)
        current_itr = 1
        with ProcessPoolExecutor(max_workers=16) as executor:
            for method in self.nonp_methods:
                history_items_recommended[method] = {}
                for exchange_point in self.exchange_points:

                    print()

                    self_id = id(self)
                    parameters = [
                        [self_id],
                        [method],
                        [exchange_point],
                        available_users,
                    ]
                    parameters = list(itertools.product(*parameters))

                    pbar = tqdm(
                        executor.map(PercentageInteraction.rec_npers, parameters),
                        total=len(parameters),
                    )
                    pbar.set_description(
                        f"{current_itr}/{itr} Stage 1 - {method} {exchange_point}"
                    )

                    results = [i for i in pbar]
                    update = [
                        rec_items
                        for result in results
                        for rec_items in result[2]
                        if rec_items[2] >= 4
                    ]

                    updated_train_dataset = Dataset(
                        data=np.concatenate((train_dataset.data, update))
                    )
                    updated_train_dataset.update_from_data()
                    updated_train_dataset.update_num_total_users_items()

                    current_model = copy.deepcopy(model)
                    current_model.reset(updated_train_dataset)
                    for [uid, item, rating, timestamp] in update:
                        current_model.observe(
                            None,
                            (uid, item),
                            rating,
                            {"vf_info": None, "asp_info": None},
                        )

                    pbar = tqdm(results, position=0, leave=True)
                    pbar.set_description(f"{current_itr}/{itr} Stage 2 - MAB Problem")

                    history_items_recommended[method][exchange_point] = [
                        self.rec_mab(result, current_model) for result in pbar
                    ]
                    current_itr += 1

        # pickle.dump(history_items_recommended, open("results_exp_ant/history_items_recommended_"+self.dataset+".pk", "wb"))
        print(f"\nCurrent Time: {time.time()-start_time:.2f} seconds")
        return history_items_recommended, None
