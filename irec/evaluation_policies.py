from collections import defaultdict
from tabulate import tabulate
from scipy.sparse import base

from threadpoolctl import threadpool_limits
import scipy.sparse
import numpy as np
import random
from tqdm import tqdm
import irec.value_functions
import irec.value_functions.MostPopular
import irec.value_functions.OurMethodInit
from irec.CandidateActions import OneUserCandidateActions
from irec.utils.dataset import Dataset
from irec.agents import Agent
from irec.utils.utils import print_dict

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import os

"""Evaluation Policies.

This module implements several assessment policies that will define how 
to assess the agent after the recommendation process.
"""

class EvaluationPolicy:
    """EvaluationPolicy.
        
    Defines a form of evaluation for the recommendation process.
    """

    def evaluate(
        self, model: Agent, train_dataset: Dataset, test_dataset: Dataset
    ) -> [list, dict]:
        """evaluate.
        
        Performs the form of evaluation according to the chosen policy.

        Args:
            model (Agent): model
            train_dataset (Dataset): train_dataset
            test_dataset (Dataset): test_dataset

        Returns:
            Tuple[list, dict]:
        """
        pass


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
            _intervals = num_trials // 100
            # _intervals = num_trials // num_trials
            _num_interactions = 0
            pbar = tqdm(total=num_trials)
            pbar.set_description(f"{model.name}")
            acts_info = []

            train_consumption_matrix = scipy.sparse.csr_matrix(
                (
                    train_dataset.data[:, 2],
                    (train_dataset.data[:, 0], train_dataset.data[:, 1]),
                ),
                (train_dataset.num_total_users, train_dataset.num_total_items),
            )

            items_popularity = irec.value_functions.MostPopular.MostPopular.get_items_popularity(
                train_consumption_matrix, normalize=True
            )

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
                    info['rec_items']=actions[1]
                    if False and isinstance(model.value_function,irec.value_functions.OurMethodInit.OurMethodInit):
                        if uid == 4653:
                            info['popularity_correlation']=scipy.stats.pearsonr(items_popularity[items_not_recommended],info['vf_info']['items_score'])[0]
                            info['popularity_percentile']= scipy.stats.percentileofscore(items_popularity,items_popularity[actions[1][0]])
                            print('------ interaction',info["user_interaction"])
                            print_dict(info)
                            print('------')
                        del info['vf_info']['items_score']
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
                if i % _intervals == 0 and i != 0:
                    pbar.update(_num_interactions)
                    _num_interactions = 0
            pbar.update(_num_interactions)
            _num_interactions = 0
            pbar.close()
            return history_items_recommended, acts_info


class InteractionSample(EvaluationPolicy):
    def __init__(self, num_interactions, interaction_size, rseed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_interactions = num_interactions
        self.interaction_size = interaction_size
        self.rseed = rseed

    def evaluate(self, model, train_dataset, test_dataset):
        np.random.seed(self.rseed)
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

        data = np.vstack((train_dataset.data, test_dataset.data))
        from irec.utils.dataset import Dataset

        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()

        consumption_matrix = scipy.sparse.csr_matrix(
            (dataset.data[:, 2], (dataset.data[:, 0], dataset.data[:, 1])),
            (dataset.num_total_users, dataset.num_total_items),
        )

        num_users_to_sample = 6
        num_consumed = (test_consumption_matrix > 0).sum(axis=1).A.flatten()
        users_selected = []
        num_users_to_sample -= len(users_selected)
        uids = np.nonzero(num_consumed >= 100)[0]
        users_sampled = np.random.choice(uids, num_users_to_sample, replace=False)
        users_selected.extend(users_sampled)

        users_items_recommended = defaultdict(list)
        num_users_selected = len(users_selected)
        print(f"Starting {model.name} Training")
        model.reset(train_dataset)
        print(f"Ended {model.name} Training")
        users_num_interactions = defaultdict(int)
        available_users = set(users_selected)

        history_items_recommended = []

        num_trials = num_users_selected * self.num_interactions
        _intervals = num_trials // 20
        _num_interactions = 0
        pbar = tqdm(total=num_trials)
        pbar.set_description(f"{model.name}")
        for i in range(num_trials):
            uid = random.sample(available_users, k=1)[0]
            not_recommended = np.ones(num_total_items, dtype=bool)
            not_recommended[users_items_recommended[uid]] = 0
            items_not_recommended = np.nonzero(not_recommended)[0]
            items_score, info = model.action_estimates((uid, items_not_recommended))
            best_items = items_not_recommended[
                np.argpartition(items_score, -self.interaction_size)[
                    -self.interaction_size :
                ]
            ]
            users_items_recommended[uid].extend(best_items)

            for item in best_items:
                history_items_recommended.append((uid, item))
                model.observe(
                    None, (uid, item), test_consumption_matrix[uid, item], info
                )
            model.increment_time()
            users_num_interactions[uid] += 1
            if users_num_interactions[uid] == self.num_interactions:
                available_users = available_users - {uid}

            _num_interactions += 1
            if i % _intervals == 0 and i != 0:
                pbar.update(_num_interactions)
                _num_interactions = 0

        pbar.update(_num_interactions)
        _num_interactions = 0
        pbar.close()
        items_entropy = irec.value_functions.Entropy.get_items_entropy(
            consumption_matrix
        )
        items_popularity = irec.value_functions.MostPopular.get_items_popularity(
            consumption_matrix, normalize=False
        )
        for uid, items in users_items_recommended.items():

            colors = mpl.cm.rainbow(np.linspace(0, 1, len(items)))
            fig = plt.figure(figsize=(8, 5))
            # plt.colorbar(colors)
            plt.rcParams.update({"font.size": 14})
            plt.scatter(items_entropy, items_popularity, s=100, color="#d1d1d1")

            plt.scatter(
                items_entropy[items], items_popularity[items], s=100, color=colors
            )

            plt.text(
                0,
                1.5,
                "Correlation: %.4f"
                % scipy.stats.pearsonr(items_entropy, items_popularity)[0],
                bbox={"facecolor": "red", "alpha": 0.5, "pad": 5},
            )
            plt.xlabel("Entropy")
            plt.ylabel("Popularity")
            fig.savefig(
                os.path.join(DirectoryDependent().DIRS["img"], f"plot_{uid}.png"),
                bbox_inches="tight",
            )

        return history_items_recommended


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
                # items_score, info = model.action_estimates(
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


class OneInteraction(EvaluationPolicy):
    def __init__(self, num_interactions, interaction_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_interactions = num_interactions
        self.interaction_size = interaction_size

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
            _intervals = num_trials // 20
            _num_interactions = 0
            pbar = tqdm(total=num_trials)
            pbar.set_description(f"{model.name}")
            no_items_recommended_users = set(test_users)

            train_consumption_matrix = scipy.sparse.csr_matrix(
                (
                    train_dataset.data[:, 2],
                    (train_dataset.data[:, 0], train_dataset.data[:, 1]),
                ),
                (train_dataset.num_total_users, train_dataset.num_total_items),
            )

            items_entropy = irec.value_functions.Entropy.get_items_entropy(
                train_consumption_matrix
            )
            items_popularity = irec.value_functions.MostPopular.get_items_popularity(
                train_consumption_matrix, normalize=True
            )
            del train_consumption_matrix
            # correlations = defaultdict(dict)
            items_value = defaultdict(dict)
            items_value_table = []
            membership = defaultdict(dict)
            # membership_table = [['Dataset','Method','Pop. Corr.','Ent. Corr.']]
            membership_table = []
            top_k_nonp = 100
            top_k_nonp_items_popularity = np.argsort(items_popularity)[::-1][
                :top_k_nonp
            ]
            top_k_nonp_items_entropy = np.argsort(items_entropy)[::-1][:top_k_nonp]

            for i in range(num_trials):
                uid = random.sample(available_users, k=1)[0]
                not_recommended = np.ones(num_total_items, dtype=bool)
                not_recommended[users_items_recommended[uid]] = 0
                items_not_recommended = np.nonzero(not_recommended)[0]
                items_score, info = model.action_estimates((uid, items_not_recommended))
                best_items = items_not_recommended[
                    np.argpartition(items_score, -self.interaction_size)[
                        -self.interaction_size :
                    ]
                ]
                users_items_recommended[uid].extend(best_items)

                for item in best_items:
                    history_items_recommended.append((uid, item))
                    model.observe(
                        None, (uid, item), test_consumption_matrix[uid, item], info
                    )
                model.increment_time()

                users_num_interactions[uid] += 1
                if users_num_interactions[uid] == self.num_interactions:
                    available_users = available_users - {uid}

                if users_num_interactions[uid] == 1:
                    # correlations['Popularity'][uid]= scipy.stats.pearsonr(items_score,items_popularity)[0]
                    # correlations['Entropy'][uid]= scipy.stats.pearsonr(items_score,items_entropy)[0]
                    items_value["Popularity"][uid] = np.mean(
                        items_popularity[best_items]
                    )
                    items_value["Entropy"][uid] = np.mean(items_entropy[best_items])
                    membership["Popularity"][uid] = len(
                        set(best_items).intersection(set(top_k_nonp_items_popularity))
                    )
                    membership["Entropy"][uid] = len(
                        set(best_items).intersection(set(top_k_nonp_items_entropy))
                    )
                    no_items_recommended_users -= {uid}

                _num_interactions += 1
                if i % _intervals == 0 and i != 0:
                    pbar.update(_num_interactions)
                    _num_interactions = 0

                if len(no_items_recommended_users) == 0:
                    print(f"Finished recommendations in {i+1} trial")
                    base_name = "None"
                    if train_dataset.num_total_users == 69878:
                        base_name = "MovieLens 10M"
                    elif train_dataset.num_total_users == 53423:
                        base_name = "Good Books"
                    elif train_dataset.num_total_users == 15400:
                        base_name = "Yahoo Music"
                    # items_value_table.append(base_name,model.name,*[np.mean(list(corr_values.values())) for corr_name, corr_values in items_value.items()])
                    # membership_table.append(base_name,model.name,*[np.mean(list(corr_values.values())) for corr_name, corr_values in membership.items()])
                    # print('Correlation: {} {} {} {}'.format(train_dataset.num_total_users,train_dataset.num_total_items,model.name,
                    # ' '.join(['{} {}'.format(corr_name,np.mean(list(corr_values.values()))) for corr_name, corr_values in correlations.items() ])))
                    # print('Items_Values: {} {} {} {}'.format(train_dataset.num_total_users,train_dataset.num_total_items,model.name,
                    # ' '.join(['{} {}'.format(corr_name,np.mean(list(corr_values.values()))) for corr_name, corr_values in items_value.items() ])
                    # ))
                    print(
                        "Items_Values: {} {} {} {}".format(
                            train_dataset.num_total_users,
                            train_dataset.num_total_items,
                            model.name,
                            " ".join(
                                [
                                    "{} {}".format(
                                        corr_name, np.mean(list(corr_values.values()))
                                    )
                                    for corr_name, corr_values in items_value.items()
                                ]
                            ),
                        )
                    )
                    print(
                        "Membership: {} {} {} {}".format(
                            train_dataset.num_total_users,
                            train_dataset.num_total_items,
                            model.name,
                            " ".join(
                                [
                                    "{} {}".format(
                                        corr_name, np.mean(list(corr_values.values()))
                                    )
                                    for corr_name, corr_values in membership.items()
                                ]
                            ),
                        )
                    )
                    break

            # print(tabulate(items_value_table, tablefmt='psql'))
            # print(tabulate(membership_table, tablefmt='psql'))
            pbar.update(_num_interactions)
            _num_interactions = 0
            pbar.close()
            return history_items_recommended
