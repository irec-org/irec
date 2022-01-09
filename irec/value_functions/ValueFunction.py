from irec.ActionCollection import ActionCollection
from irec.Action import Action
from typing import Any, Dict, Tuple


class ValueFunction:
    """ValueFunction.
    One of the goals in RS is to estimate the usefulness of a recommendation for a user.
    From an RL perspective, in addition to an estimate of the immediate utility/reward of
    an action, the agent aims to learn from experience the long-term value of an action.
    The action value function Q(s, a) defines the value of performing an action a in state s.
    """

    def __init__(self, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
        """
        del args, kwargs

    def reset(self, observation: Any):
        """reset.

        Reset all model attributes.

        Args:
            observation (Any): observation
        """
        train_dataset = observation
        pass
        # super().reset(train_dataset)

    def action_estimates(
        self, candidate_actions: ActionCollection
    ) -> Tuple[Any, Dict[Any, Any]]:
        """action_estimates.

        Received as input the currently available actions the agent can perform and returns
        the necessary information, limited to the actions the agent can perform at time step t,
        to allow a specific next action to be performed.

        Args:
            candidate_actions (ActionCollection): candidate_actions

        Returns:
            Tuple[Any, dict]: Scores and information
        """

        # uid = candidate_actions[0]
        # candidate_items = candidate_actions[1]
        return None, dict()

    def update(
        self, observation: Any, action: Action, reward: float, info: dict
    ) -> None:
        """update.

        According to the observations retrieved at each time step, the agent updates its
        estimate of the value function based on the reward received.

        Args:
            observation (Any): observation
            action (Action): action
            reward (float): reward
            info (dict): info
        """
        # uid = action[0]
        # item = action[1]
        # additional_data = info
        return None
