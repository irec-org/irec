from .base import ActionSelectionPolicy
import numpy as np

class ASPGenericGreedy(ActionSelectionPolicy):

    """ASPGenericGreedy
    
        
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_actions(self, actions, action_estimates, actions_num):
        actions = actions[
            np.argpartition(action_estimates, -actions_num)[-actions_num:]
        ]
        return actions, None

    def update(self, observation, action, reward, info):
        pass

    def reset(self, observation):
        pass
