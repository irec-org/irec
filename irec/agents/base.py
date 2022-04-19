from .action_selection_policies.base import ActionSelectionPolicy
from .value_functions.base import ValueFunction
from irec.agents.action import Action
from typing import Dict, Any

"""Recommender System Agent.

An agent can be defined as the Recommendation System (RS) itself. At each time step,
the Agent requires a recommendation (that is, performs an action) given a context or
state of the RS Environment. After offering a recommendation, either an item or a list of items,
the Agent will receive a reward from the Environment. In the RS field, rewards are usually
a scalar value representative of the user's explicit/implicit feedback.
"""


class Agent:
    """Agent.
    The agent's goal is to learn a policy, a strategy that indicates the correct action to be taken
    in each state/situation, in order to maximize the total reward collected throughout the interaction.

    Every Agent has a value function, whose main task is to estimate the usefulness of a
    recommendation for a user. From an RL perspective, in addition to an estimate of the
    immediate utility/reward of an action, the agent aims to learn from experience the long-term
    value of an action. In short, the value function incorporates the agent's goals, quantifying
    the expected long-term consequences of decisions.

    In addition, the action selection policy uses the information provided by the value function.
    to select the next action to be performed.
    """

    def __init__(
        self,
        value_function: ValueFunction,
        action_selection_policy: ActionSelectionPolicy,
        name: str,
        *args,
        **kwargs
    ):

        """__init__.

        Args:
            value_function (ValueFunction): value_function
            action_selection_policy (ActionSelectionPolicy): action_selection_policy
            name (str): name
            args:
            kwargs:
        """
        super().__init__(*args, **kwargs)
        self.value_function: ValueFunction = value_function
        self.action_selection_policy = action_selection_policy
        self.name = name

    def act(self, candidate_actions: Action, actions_num: int):
        """act.

        An action is a recommendation, which will be made about the items available to a particular target user.

        Args:
            candidate_actions (Action): candidate_actions
            actions_num (int): actions_num
        """
        raise NotImplementedError

    def observe(
        self, observation: Any, action: Action, reward: float, info: Dict[Any, Any]
    ):
        """observe.

        After each action, the agent receives a reward, and determines what kind of information the agent
        can observe to update its value function and its selection policy.

        Args:
            observation (Any): observation
            action (Action): action
            reward (float): reward
            info (dict): info
        """

        raise NotImplementedError

    def reset(self, observation: Any):
        """reset.

        Reset agent parameters.

        Args:
            observation (Any): observation
        """

        raise NotImplementedError