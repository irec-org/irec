from .base import ActionSelectionPolicy
import numpy as np

class ASPGreedy(ActionSelectionPolicy):

    """ASPGreedy
    
        Will always select the best items available
    """

    def select_actions(self, actions, actions_estimate, actions_num):
        return (
            actions[0],
            actions[1][np.argpartition(actions_estimate, -actions_num)[-actions_num:]],
        ), None

    def update(self, observation, action, reward, info):
        pass

    def reset(self, observation):
        pass
