import numpy as np
from collections import defaultdict
import lib.value_functions.Entropy


class ActionSelectionPolicy:

    def __init__(self, *args, **kwargs):
        pass

    def select_actions(self, actions, action_estimates, actions_num):
        raise NotImplementedError

    def update(self, observation, action, reward, info):
        raise NotImplementedError

    def reset(self, observation):
        raise NotImplementedError


class ASPGreedy(ActionSelectionPolicy):

    def select_actions(self, actions, action_estimates, actions_num):
        return (actions[0],
                actions[1][np.argpartition(action_estimates,
                                           -actions_num)[-actions_num:]]), None

    def update(self, observation, action, reward, info):
        pass

    def reset(self, observation):
        pass


class ASPEGreedy(ActionSelectionPolicy):

    def __init__(self, epsilon, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.epsilon = epsilon

    def select_actions(self, actions, action_estimates, actions_num):
        greedy_actions = np.argpartition(action_estimates,
                                         -actions_num)[-actions_num:][::-1]
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
            return (actions[0], actions[1][np.argpartition(
                action_estimates, -actions_num)[-actions_num:]]), None

        top_estimates_index_actions = np.argpartition(
            action_estimates,
            -self.input_filter_size)[-self.input_filter_size:][::-1]

        actions = (actions[0], actions[1][top_estimates_index_actions])
        action_estimates = action_estimates[top_estimates_index_actions]

        rule_action_estimates = self.rule.action_estimates(actions)[0]
        top_rule_index_actions = np.argpartition(
            rule_action_estimates, -actions_num)[-actions_num:][::-1]
        return (actions[0], actions[1][top_rule_index_actions]), None

    def update(self, observation, action, reward, info):
        if reward >= 4:
            self.users_num_consumption[action[0]] += 1
        self.rule.update(observation, action, reward, info)

    def reset(self, observation):
        self.users_num_consumption = defaultdict(int)
        self.rule.reset(observation)


class ASPGenericGreedy(ActionSelectionPolicy):

    def select_actions(self, actions, action_estimates, actions_num):
        # print(actions,action_estimates)
        return actions[np.argpartition(action_estimates,
                                       -actions_num)[-actions_num:]], None

    def update(self, observation, action, reward, info):
        pass

    def reset(self, observation):
        pass
