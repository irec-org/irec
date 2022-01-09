from . import DatasetLoader
import irec.utils.dataset as dataset
import irec.agents


class Factory:
    def __init__(self) -> None:
        pass

    def create(self, name, **parameters):
        raise NotImplementedError()


class ActionSelectionPolicyFactory(Factory):
    def __init__(self, value_function_factory) -> None:
        self.value_function_factory = value_function_factory
        pass

    def create(self, action_selection_policy_settings):
        import irec.action_selection_policies

        action_selection_policy_name = list(action_selection_policy_settings.keys())[0]
        action_selection_policy_parameters = list(
            action_selection_policy_settings.values()
        )[0]

        action_selection_policy = eval(
            "irec.action_selection_policies." + action_selection_policy_name
        )(**action_selection_policy_parameters)

        if isinstance(
            action_selection_policy, irec.action_selection_policies.ASPReranker
        ):
            action_selection_policy.rule = self.value_function_factory.create(
                action_selection_policy.rule
            )
        return action_selection_policy


class ValueFunctionFactory(Factory):
    def create(self, value_function_settings):
        value_function_name = list(value_function_settings.keys())[0]
        value_function_parameters = list(value_function_settings.values())[0]
        value_function = None
        if value_function_name in [
            "OurMethodRandom",
            "OurMethodRandPopularity",
            "OurMethodEntropy",
            "OurMethodPopularity",
            "OurMethodOne",
            "OurMethodZero",
        ]:
            exec("import irec.value_functions.WSPBInit")
            value_function = eval(
                "irec.value_functions.WSPBInit.{}".format(value_function_name)
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
    def __init__(
        self,
        value_function_factory=ValueFunctionFactory(),
        action_selection_policy_factory=ActionSelectionPolicyFactory(
            ValueFunctionFactory()
        ),
    ) -> None:
        self.value_function_factory = value_function_factory
        self.action_selection_policy_factory = action_selection_policy_factory
        pass

    def create(self, agent_name, agent_settings):
        agent_class_name = list(agent_settings.keys())[0]
        agent_parameters = list(agent_settings.values())[0]
        agent_class = eval("irec.agents." + agent_class_name)

        agent_class_parameters = {}
        action_selection_policy = self.action_selection_policy_factory.create(
            agent_parameters["action_selection_policy"]
        )
        value_function = self.value_function_factory.create(
            agent_parameters["value_function"]
        )
        agents = []
        if agent_name in [
            "NaiveEnsemble",
            "TSEnsemble_Pop",
            "TSEnsemble_PopEnt",
            "TSEnsemble_Entropy",
            "TSEnsemble_Random",
        ]:
            for _agent in agent_parameters["agents"]:
                sub_agent_settings = list(_agent.values())[0]
                # sub_agent_class_name = list(sub_agent_settings.keys())[0]
                # sub_agent_parameters = list(sub_agent_settings.values())[0]
                new_agent = self.create(list(_agent.keys())[0], sub_agent_settings)
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


class DatasetLoaderFactory(Factory):
    def create(self, dataset_name, dataset_settings):
        dataset_class_name = list(dataset_settings.keys())[0]
        dataset_parameters = list(dataset_settings.values())[0]
        dataset_class = eval("irec.utils.DatasetLoader." + dataset_class_name)
        return dataset_class(**dataset_settings[dataset_class_name])
