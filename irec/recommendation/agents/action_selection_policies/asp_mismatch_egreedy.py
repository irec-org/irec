from typing import Any, Iterable, Tuple
from collections import defaultdict
import numpy as np
import scipy

from irec.environment.dataset import Dataset
from irec.recommendation.agents.base import ActionSelectionPolicy
from irec.recommendation.agents.value_functions.most_popular import MostPopular
from irec.recommendation.agents.value_functions.entropy import Entropy
from irec.recommendation.agents.value_functions.log_pop_ent import LogPopEnt


class ASPMismatchEgreedy(ActionSelectionPolicy):
    def __init__(self, epsilon: float, k: int, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.epsilon = epsilon
        self.k = k
        self.miss_per_uid = defaultdict(int)
        self.rank: np.ndarray = None

    def select_actions(self, actions, actions_estimate, num_actions):
        uid = actions[0]

        # M Numero de erros consecutivos
        # T Threshold (k no código)

        if self.miss_per_uid[uid] >= self.k and self.k != 0:
            self.miss_per_uid[uid] = 0
            actions_indexes = self.exploit(actions, self.rank[actions[1]], num_actions)
        elif self.epsilon < np.random.rand():
            actions_indexes = self.exploit(actions, actions_estimate, num_actions)
        else:
            actions_indexes = self.explore(actions, actions_estimate, num_actions)

        return (uid, actions[1][actions_indexes]), None

    def exploit(self, actions, actions_estimate, num_actions) -> Iterable[int]:
        greedy_actions = np.argpartition(actions_estimate, -num_actions)[-num_actions:][
            ::-1
        ]
        actions_indexes = list()
        for i in range(num_actions):
            j = 0
            while True:
                action_index = greedy_actions[j]
                if action_index not in actions_indexes:
                    break
                j += 1
            actions_indexes.append(action_index)
        return actions_indexes

    def explore(self, actions, actions_estimate, num_actions) -> Iterable[int]:
        actions_indexes = list()
        for i in range(num_actions):
            while True:
                action_index = np.random.randint(len(actions_estimate))
                if action_index not in actions_indexes:
                    break
            actions_indexes.append(action_index)
        return actions_indexes

    def update(self, observation: Any, action: Tuple[int, int], reward: float, info: Any):
        uid = action[0]

        if reward == 0:
            self.miss_per_uid[uid] += 1
        else:
            self.miss_per_uid[uid] = 0

    def reset(self, observation: Dataset):
        m = scipy.sparse.csr_matrix(
            ( observation.data[:, 2], (observation.data[:, 0], observation.data[:, 1]) ),
            ( observation.num_total_users, observation.num_total_items )
        )
        items_popularity = MostPopular.get_items_popularity(m, normalize=False)
        items_entropy = Entropy.get_items_entropy(m)
        self.rank = LogPopEnt.get_items_logpopent(items_popularity, items_entropy)
        # self.rank = items_entropy * items_popularity
