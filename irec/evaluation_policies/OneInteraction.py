from irec.agents.value_functions.MostPopular import MostPopular
from irec.agents.value_functions.BestRated import BestRated
from irec.agents.value_functions.LogPopEnt import LogPopEnt
from irec.agents.value_functions.Entropy import Entropy
from .EvaluationPolicy import EvaluationPolicy
from threadpoolctl import threadpool_limits
from irec.environment.dataset import Dataset
from collections import defaultdict
from irec.agents.base import Agent
from tqdm import tqdm
import scipy.sparse
import scipy.stats
import numpy as np
import random

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

            items_entropy = Entropy.get_items_entropy(
                train_consumption_matrix
            )
            items_popularity = MostPopular.get_items_popularity(
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

            pbar.update(_num_interactions)
            _num_interactions = 0
            pbar.close()
            return history_items_recommended
