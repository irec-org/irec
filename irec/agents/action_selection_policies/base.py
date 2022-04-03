from irec.agents.ActionCollection import ActionCollection

class ActionSelectionPolicy:
    def __init__(self, *args, **kwargs):
        pass

    def select_actions(self, actions: ActionCollection, action_estimates, actions_num):
        raise NotImplementedError

    def update(self, observation, action, reward, info):
        raise NotImplementedError

    def reset(self, observation):
        raise NotImplementedError