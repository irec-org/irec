class DotDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
class Agent:
    def __init__(self,value_function,action_selection_policy):
        self.value_function = value_function
        self.action_selection_policy = action_selection_policy
        self.state = DotDict()
    def act(self):
        raise NotImplementedError
    def observe(self,observation,action,reward,done):
        raise NotImplementedError
    def reset(self,observation):
        raise NotImplementedError

class SimpleAgent:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def act(self,candidate_actions,actions_num):
        action_estimates,vf_info = self.value_function.action_estimates(candidate_actions)
        actions_indexes,asp_info = self.action_selection_policy.select_actions(action_estimates,actions_num)
        actions = candidate_actions[actions_indexes]
        return actions, {'vf_info':vf_info,'asp_info':asp_info}
    def observe(self,observation,action,reward,done,info):
        self.value_function.update(observation,action,reward,info['vf_info'])
        self.action_selection_policy.update(observation,action,reward,info['asp_info'])
        return None
    def reset(self,observation):
        if observation != None:
            self.value_function.reset(observation)
            self.action_selection_policy.reset(observation)
            return None
        else:
            raise NotImplementedError

