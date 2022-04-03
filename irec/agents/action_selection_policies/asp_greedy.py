from .base import ActionSelectionPolicy
import numpy as np

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
