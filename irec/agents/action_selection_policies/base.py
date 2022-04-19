from irec.agents.action import Action


class ActionSelectionPolicy:

    """Action Selection Policy.

    The action selection policy represents the policy used by the agent
    to choose one (or more) items to be recommended. In general, such 
    policies have two competing objectives: exploiting the items with 
    the highest rewards in the past; or exploring unknown items to 
    improve the systems knowledge.
    """

    def __init__(self, *args, **kwargs):
        pass

    def select_actions(self, actions: Action, actions_estimate, actions_num: int):
        """select actions
        
            Select the best actions (recommendations)

        Args:
            actions: all candidate actions (items) to be recommended
            actions_estimates: action estimates by value function
            actions_num: number of actions that will be returned

        Return:
            actions: the actions (items) returned
            info: additional Information
        """

        raise NotImplementedError

    def update(self, observation, action, reward, info):
        """update

        Update of actions selection policy information

        Args:
            action: tuple (uid, iid)
            reward: rating
            info: additional Information
        """

        raise NotImplementedError

    def reset(self, observation):
        """reset

        Reset all action selection policy attributes

        """

        raise NotImplementedError