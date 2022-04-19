from .base import ActionSelectionPolicy
from collections import defaultdict
import numpy as np

class ASPReranker(ActionSelectionPolicy):

    """ASPReranker

    
    """

    def __init__(self, rule, input_filter_size, rerank_limit, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.rule = rule
        self.input_filter_size = input_filter_size
        self.rerank_limit = rerank_limit

    def select_actions(self, actions, actions_estimate, actions_num):
        if self.users_num_consumption[actions[0]] > self.rerank_limit:
            return (
                actions[0],
                actions[1][
                    np.argpartition(actions_estimate, -actions_num)[-actions_num:]
                ],
            ), None

        top_estimates_index_actions = np.argpartition(
            actions_estimate, -self.input_filter_size
        )[-self.input_filter_size :][::-1]

        actions = (actions[0], actions[1][top_estimates_index_actions])
        actions_estimate = actions_estimate[top_estimates_index_actions]

        rule_actions_estimate = self.rule.actions_estimate(actions)[0]
        top_rule_index_actions = np.argpartition(rule_actions_estimate, -actions_num)[
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
