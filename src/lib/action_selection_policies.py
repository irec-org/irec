import numpy as np


class ActionSelectionPolicy:

    def __init__(self,*args,**kwargs):
        pass

    def select_actions(self, action_estimates, actions_num):
        raise NotImplementedError

    def update(self, observation, action, reward, info):
        raise NotImplementedError

    def reset(self, observation):
        raise NotImplementedError


class ASPGreedy(ActionSelectionPolicy):

    def select_actions(self, action_estimates, actions_num):
        return np.argpartition(action_estimates,
                               -actions_num)[-actions_num:], None

    def update(self, observation, action, reward, info):
        pass

    def reset(self, observation):
        pass


class ASPEGreedy(ActionSelectionPolicy):

    def __init__(self, epsilon, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.epsilon = epsilon

    def select_actions(self, action_estimates, actions_num):
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

        return actions_indexes, None

    def update(self, observation, action, reward, info):
        pass

    def reset(self, observation):
        pass
