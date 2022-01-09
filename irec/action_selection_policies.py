from typing import Any
from irec.ActionCollection import ActionCollection
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict

from irec.value_functions.MostPopular import MostPopular
from irec.value_functions.Entropy import Entropy
import scipy.sparse
from irec.value_functions.LogPopEnt import LogPopEnt
import random
import mf


class ActionSelectionPolicy:
    def __init__(self, *args, **kwargs):
        pass

    def select_actions(self, actions: ActionCollection, action_estimates, actions_num):
        raise NotImplementedError

    def update(self, observation, action, reward, info):
        raise NotImplementedError

    def reset(self, observation):
        raise NotImplementedError


class ASPGreedy(ActionSelectionPolicy):
    def select_actions(self, actions, action_estimates, actions_num):
        return (
            actions[0],
            actions[1][np.argpartition(action_estimates, -actions_num)[-actions_num:]],
        ), None

    def update(self, observation, action, reward, info):
        pass

    def reset(self, observation):
        pass


class ASPEGreedy(ActionSelectionPolicy):
    def __init__(self, epsilon, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.epsilon = epsilon

    def select_actions(self, actions, action_estimates, actions_num):
        greedy_actions = np.argpartition(action_estimates, -actions_num)[-actions_num:][
            ::-1
        ]
        actions_indexes = []
        for i in range(actions_num):
            if self.epsilon < np.random.rand():
                j = 0
                while True:
                    action_index = greedy_actions[j]
                    if action_index not in actions_indexes:
                        break
                    j += 1
                actions_indexes.append(action_index)
            else:
                while True:
                    action_index = np.random.randint(len(action_estimates))
                    if action_index not in actions_indexes:
                        break
                actions_indexes.append(action_index)

        return (actions[0], actions[1][actions_indexes]), None

    def update(self, observation, action, reward, info):
        pass

    def reset(self, observation):
        pass


class ASPReranker(ActionSelectionPolicy):
    def __init__(self, rule, input_filter_size, rerank_limit, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.rule = rule
        self.input_filter_size = input_filter_size
        self.rerank_limit = rerank_limit

    def select_actions(self, actions, action_estimates, actions_num):
        if self.users_num_consumption[actions[0]] > self.rerank_limit:
            return (
                actions[0],
                actions[1][
                    np.argpartition(action_estimates, -actions_num)[-actions_num:]
                ],
            ), None

        top_estimates_index_actions = np.argpartition(
            action_estimates, -self.input_filter_size
        )[-self.input_filter_size :][::-1]

        actions = (actions[0], actions[1][top_estimates_index_actions])
        action_estimates = action_estimates[top_estimates_index_actions]

        rule_action_estimates = self.rule.action_estimates(actions)[0]
        top_rule_index_actions = np.argpartition(rule_action_estimates, -actions_num)[
            -actions_num:
        ][::-1]
        return (actions[0], actions[1][top_rule_index_actions]), None

    def update(self, observation, action, reward, info):
        if reward >= 4:
            self.users_num_consumption[action[0]] += 1
        self.rule.update(observation, action, reward, info)

    def reset(self, observation):
        self.users_num_consumption = defaultdict(int)
        self.rule.reset(observation)


class ASPGenericGreedy(ActionSelectionPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.save_played_actions = save_played_actions
        pass

    def select_actions(self, actions, action_estimates, actions_num):
        # print(actions,action_estimates)
        # info = {''}
        actions = actions[
            np.argpartition(action_estimates, -actions_num)[-actions_num:]
        ]
        # if self.save_played_actions:
        # info = {'played_actions': factions}
        return actions, None

    def update(self, observation, action, reward, info):
        pass

    def reset(self, observation):
        pass


class ASPICGreedy(ActionSelectionPolicy):
    def __init__(self, stop, num_clusters, num_lat, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = stop
        self.num_clusters = num_clusters
        self.num_lat = num_lat

    def choose_group(self, uid):
        b = self.users_num_consumption_groups[uid]
        return np.random.choice(np.flatnonzero(b == b.min()))

    def get_item_group(self, item):
        return self.groups_items[item]

    def select_actions(self, actions, action_estimates, actions_num):
        uid = actions[0]
        if self.users_num_consumption[uid] >= self.stop:
            return (
                actions[0],
                actions[1][
                    np.argpartition(action_estimates, -actions_num)[-actions_num:]
                ],
            ), None
        else:
            i = 0
            while True:
                i += 1
                # print(i)
                # g = self.choose_group(uid)
                # try:
                g = random.randint(0, self.num_clusters)
                g_items = np.array(self.groups_items[g])
                # print(actions[1],g_items)
                g_items = np.intersect1d(actions[1], g_items)
                # print(g_items)
                if len(g_items) == 0:
                    continue
                g_items = list(g_items)
                top_item = g_items[np.argmax(self.items_popularity[g_items])]
                break
                # break
                # except:
                # continue

            return (
                actions[0],
                [top_item],
            ), None

    def update(self, observation, action, reward, info):
        uid = action[0]
        # iid = action[1]
        self.users_num_consumption[uid] += 1
        pass

    def reset(self, observation):
        train_dataset = observation
        self.train_dataset = train_dataset

        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (
                self.train_dataset.data[:, 2],
                (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1]),
            ),
            (self.train_dataset.num_total_users, self.train_dataset.num_total_items),
        )
        mf_model = mf.SVD(num_lat=self.num_lat)
        mf_model.fit(self.train_consumption_matrix)
        self.items_weights = mf_model.items_weights
        self.num_latent_factors = len(self.items_weights[0])

        kmeans = KMeans(self.num_clusters).fit(self.items_weights)

        self.num_total_items = self.train_dataset.num_total_items
        self.num_total_users = self.train_dataset.num_total_users

        self.groups_items = defaultdict(list)
        print(len(kmeans.labels_))
        for uid, group in enumerate(kmeans.labels_):
            self.groups_items[group].append(uid)

        self.groups = list(range(self.num_clusters))
        self.users_num_consumption = defaultdict(int)
        self.users_num_consumption_groups: Any = defaultdict(
            lambda: np.zeros(self.num_clusters)
        )

        self.items_entropy = Entropy.get_items_entropy(self.train_consumption_matrix)
        self.items_popularity = MostPopular.get_items_popularity(
            self.train_consumption_matrix, normalize=False
        )
        self.items_logpopent = LogPopEnt.get_items_logpopent(
            self.items_popularity, self.items_entropy
        )
