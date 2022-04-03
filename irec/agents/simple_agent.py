from irec.agents.action_collection import ActionCollection
from .action import UIAction
from .base import Agent
from typing import Any

class SimpleAgent(Agent):
    """SimpleAgent."""

    def __init__(self, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
        """
        super().__init__(*args, **kwargs)

    def act(self, candidate_actions: ActionCollection, actions_num: int):
        """act.

        Args:
            candidate_actions (ActionCollection): candidate_actions
            actions_num (int): actions_num
        """
        action_estimates, vf_info = self.value_function.action_estimates(
            candidate_actions
        )
        actions, asp_info = self.action_selection_policy.select_actions(
            candidate_actions, action_estimates, actions_num
        )
        # actions = (candidate_actions[0],candidate_actions[1][actions_indexes])
        return actions, {"vf_info": vf_info, "asp_info": asp_info}

    def observe(self, observation: Any, action: UIAction, reward: float, info: dict):
        """observe.

        Args:
            observation (Any): observation
            action (UIAction): action
            reward (float): reward
            info (dict): info
        """
        vf_info = info.get("vf_info", None)
        asp_info = info.get("asp_info", None)
        self.value_function.update(observation, action, reward, vf_info)
        self.action_selection_policy.update(observation, action, reward, asp_info)
        return None

    def reset(self, observation: Any):
        """reset.

        Args:
            observation (Any): observation
        """
        if observation is not None:
            self.value_function.reset(observation)
            self.action_selection_policy.reset(observation)
            return None
        else:
            raise NotImplementedError