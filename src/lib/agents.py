from numpy.lib.npyio import save
from lib import value_functions
import numpy as np


class Agent:

    def __init__(self, value_function, action_selection_policy, name, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.value_function = value_function
        self.action_selection_policy = action_selection_policy
        self.name = name

    def act(self, candidate_actions, actions_num):
        raise NotImplementedError

    def observe(self, observation, action, reward, info):
        raise NotImplementedError

    def reset(self, observation):
        raise NotImplementedError


class SimpleAgent(Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(self, candidate_actions, actions_num):
        action_estimates, vf_info = self.value_function.action_estimates(
            candidate_actions)
        actions, asp_info = self.action_selection_policy.select_actions(
            candidate_actions, action_estimates, actions_num)
        # actions = (candidate_actions[0],candidate_actions[1][actions_indexes])
        return actions, {'vf_info': vf_info, 'asp_info': asp_info}

    def observe(self, observation, action, reward, info):
        vf_info = info.get('vf_info',None)
        asp_info = info.get('asp_info',None)
        self.value_function.update(observation, action, reward, vf_info)
        self.action_selection_policy.update(observation, action, reward,
                                            asp_info)
        return None

    def reset(self, observation):
        if observation != None:
            self.value_function.reset(observation)
            self.action_selection_policy.reset(observation)
            return None
        else:
            raise NotImplementedError


class SimpleEnsembleAgent(Agent):

    def __init__(self, agents, save_meta_actions=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.ensemble_method_vf = ensemble_method_vf
        self.agents = agents
        self.ensemble_candidate_actions = np.array(list(range(len(
            self.agents))))
        self.default_actions_num = 1
        self.save_meta_actions = save_meta_actions

    def act(self, candidate_actions, actions_num):
        meta_action_estimates, meta_vf_info = self.value_function.action_estimates(
            self.ensemble_candidate_actions)
        # print(m,meta_action_estimates)
        meta_actions, meta_asp_info = self.action_selection_policy.select_actions(
            self.ensemble_candidate_actions, meta_action_estimates,
            self.default_actions_num)
        meta_action = meta_actions[0]
        # print(meta_actions)
        # actions = (candidate_actions[0],candidate_actions[1][actions_indexes])
        info = {}
        selected_agent = self.agents[meta_action]
        selected_agent_actions, selected_agent_info = selected_agent.act(
            candidate_actions, actions_num)
        if meta_vf_info != None:
            info['vf_info'] = meta_vf_info
        if meta_asp_info != None:
            info['asp_info'] = meta_asp_info
        if selected_agent_info != None:
            info['selected_agent_info'] = selected_agent_info
        if self.save_meta_actions:
            info['meta_action_name'] = self.agents[meta_action].name
        if info == dict():
            info = None
        return selected_agent_actions, info

    def observe(self, observation, action, reward, info):
        vf_info = info.get('vf_info',None)
        asp_info = info.get('asp_info',None)
        self.value_function.update(observation, action, reward, vf_info)
        self.action_selection_policy.update(observation, action, reward,
                                            asp_info)
        for agent in self.agents:
            agent.observe(observation, action, reward, info)
        return None

    def reset(self, observation):
        if observation != None:
            self.value_function.reset(observation)
            self.action_selection_policy.reset(observation)
            for agent in self.agents:
                agent.reset(observation)
            return None
        else:
            raise NotImplementedError
