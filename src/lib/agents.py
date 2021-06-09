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
            candidate_actions,action_estimates, actions_num)
        # actions = (candidate_actions[0],candidate_actions[1][actions_indexes])
        return actions, {'vf_info': vf_info, 'asp_info': asp_info}

    def observe(self, observation, action, reward, info):
        self.value_function.update(observation, action, reward, info['vf_info'])
        self.action_selection_policy.update(observation, action, reward,
                                            info['asp_info'])
        return None

    def reset(self, observation):
        if observation != None:
            self.value_function.reset(observation)
            self.action_selection_policy.reset(observation)
            return None
        else:
            raise NotImplementedError
