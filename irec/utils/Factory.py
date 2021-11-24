
class Factory:
    def __init__(self) -> None:
        pass
    def create(self,name,parameters):
        raise NotImplementedError()

class ActionSelectionPolicyFactory(Factory):
    def __init__(self,value_function_factory) -> None:
        self.value_function_factory= value_function_factory
        pass

    def create(self,action_selection_policy_name,action_selection_policy_parameters):
        import irec.action_selection_policies
        action_selection_policy = eval(
                "irec.action_selection_policies." + action_selection_policy_name
                )(**action_selection_policy_parameters)

        if isinstance(action_selection_policy, irec.action_selection_policies.ASPReranker):
            action_selection_policy.rule = self.value_function_factory.create(
                    action_selection_policy.rule
                    )
        return action_selection_policy

class ValueFunctionFactory(Factory):
    def create(self,value_function_name,value_function_parameters):
        value_function = None
        if value_function_name in [
                "OurMethodRandom",
                "OurMethodRandPopularity",
                "OurMethodEntropy",
                "OurMethodPopularity",
                "OurMethodOne",
                "OurMethodZero",
                ]:
            exec("import irec.value_functions.OurMethodInit")
            value_function = eval(
                    "irec.value_functions.OurMethodInit.{}".format(value_function_name)
                    )(**value_function_parameters)
        if value_function_name in [
                "ICTRTS",
                ]:
            exec("import irec.value_functions.ICTR")
            value_function = eval(
                    "irec.value_functions.ICTR.{}".format(value_function_name)
                    )(**value_function_parameters)
        else:
            exec("import irec.value_functions.{}".format(value_function_name))
            value_function = eval(
                    "irec.value_functions.{}.{}".format(
                        value_function_name, value_function_name
                        )
                    )(**value_function_parameters)
        return value_function

class AgentFactory(Factory):
    def __init__(self,value_function_factory=ValueFunctionFactory(),action_selection_policy_factory=ActionSelectionPolicyFactory(ValueFunctionFactory())) -> None:
        self.value_function_factory= value_function_factory
        self.action_selection_policy_factory= action_selection_policy_factory
        pass
    def create(self, agent_name, agent_parameters):
        agent_class = eval("irec.agents." + agent_name)

        agent_class_parameters = {}
        action_selection_policy = self.action_selection_policy_factory.create(
                agent_parameters["action_selection_policy"]
                )
        value_function = self.value_function_factory.create(agent_parameters["value_function"])
        agents = []
        if agent_name in [
                "NaiveEnsemble",
                "TSEnsemble_Pop",
                "TSEnsemble_PopEnt",
                "TSEnsemble_Entropy",
                "TSEnsemble_Random",
                ]:
            for _agent in agent_parameters["agents"]:
                new_agent = create_agent(list(_agent.keys())[0], list(_agent.values())[0])
                agents.append(new_agent)
            agent_class_parameters["agents"] = agents
        agent_class_parameters.update(
                {
                    "action_selection_policy": action_selection_policy,
                    "value_function": value_function,
                    "name": agent_name,
                    }
                )
        return agent_class(**agent_class_parameters)
