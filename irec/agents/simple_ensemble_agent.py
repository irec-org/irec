from irec.agents.action_collection import ActionCollection
from .action import UIAction
from .base import Agent
from typing import Any, List
import numpy as np

class SimpleEnsembleAgent(Agent):
    """SimpleEnsembleAgent.
    
    iRec also allows agents based on ensemble strategies by using the tag 
    EnsembleAgent and, in this case, more than one Action Selection Policy
    and/or more than one ValueFunction can be set.
    """

    def __init__(
        self,
        agents: List[Agent],
        use_name_meta_actions: bool = True,
        save_meta_actions: bool = True,
        *args,
        **kwargs
    ):
        """__init__.

        Args:
            agents (List[Agent]): agents
            use_name_meta_actions (bool): use_name_meta_actions
            save_meta_actions (bool): save_meta_actions
            args:
            kwargs:
        """

        super().__init__(*args, **kwargs)
        self.agents = agents
        self.use_name_meta_actions = use_name_meta_actions
        if self.use_name_meta_actions:
            self.ensemble_candidate_actions = np.array([i.name for i in self.agents])
            self.actions_name_object_map = {
                act_name: self.agents[i]
                for i, act_name in enumerate(self.ensemble_candidate_actions)
            }
        else:
            self.ensemble_candidate_actions = np.array(list(range(len(self.agents))))
        self.default_actions_num = 1
        self.save_meta_actions = save_meta_actions

    def act(self, candidate_actions: ActionCollection, actions_num: int):
        """act.

        Args:
            candidate_actions (ActionCollection): candidate_actions
            actions_num (int): actions_num
        """
        info = {}
        meta_actions_estimate, meta_vf_info = self.value_function.actions_estimate(
            self.ensemble_candidate_actions
        )
        meta_actions, meta_asp_info = self.action_selection_policy.select_actions(
            self.ensemble_candidate_actions,
            meta_actions_estimate,
            self.default_actions_num,
        )
        meta_action = meta_actions[0]
        if self.use_name_meta_actions:
            selected_agent = self.actions_name_object_map[meta_action]
        else:
            selected_agent = self.agents[meta_action]
        selected_agent_actions, selected_agent_info = selected_agent.act(
            candidate_actions, actions_num
        )
        if meta_vf_info is not None:
            info["vf_info"] = meta_vf_info
        if meta_asp_info is not None:
            info["asp_info"] = meta_asp_info
        if selected_agent_info is not None:
            info["selected_agent_info"] = selected_agent_info
        if self.save_meta_actions:
            if self.use_name_meta_actions:
                info["meta_action_name"] = meta_action
                info["mai"] = meta_action
            else:
                info["meta_action_name"] = self.agents[meta_action].name
                info["mai"] = meta_action
        if info == dict():
            info = None
        return selected_agent_actions, info

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
        self.value_function.update(observation, info["mai"], reward, vf_info)
        self.action_selection_policy.update(observation, info["mai"], reward, asp_info)
        for agent in self.agents:
            agent.observe(observation, action, reward, info)
        return None

    def reset(self, observation: Any):
        """reset.

        Args:
            observation (Any): observation
        """
        if observation is not None:
            self.value_function.reset(observation)
            self.action_selection_policy.reset(observation)
            for agent in self.agents:
                agent.reset(observation)
            return None
        else:
            raise NotImplementedError
