from __future__ import annotations
from typing import List

def _import_class(root:str,
              module_name:str,
              class_name:str):

    exec(f"from {root}.{module_name} import {class_name}")

    return eval(class_name)

class AgentRegistry:
    
    from irec.agents.base import Agent
    from irec.agents.simple_agent import SimpleAgent
    from irec.agents.simple_ensemble_agent import SimpleEnsembleAgent

    _agent = {
        "Agent": Agent,
        "SimpleAgent": SimpleAgent,
        "SimpleEnsembleAgent": SimpleEnsembleAgent,
    }

    @classmethod
    def all(cls: AgentRegistry) -> List[str]:
        return list(cls._agent.keys())

    @classmethod
    def get(cls: AgentRegistry, name: str):
        return cls._agent[name]


class VFRegistry:
    

    _vf = {
        "BestRated": "best_rated",
        "CB": "cb",
        "COFIBA": "cofiba",
        "EGreedy": "e_greedy",
        "Entropy": "entropy",
        "Entropy0": "entropy0",
        "ExperimentalValueFunction": "experimental_valueFunction",
        "GenericThompsonSampling": "generic_thompson_sampling",
        "GLM_UCB": "glm_ucb",
        "HELF": "helf",
        "ICF": "icf",
        "ICTRTS": "ictr",
        "kNNBandit": "knn_bandit",
        "LinEGreedy": "lin_egreedy",
        "LinUCB": "lin_ucb",
        "LinearEGreedy": "linear_egreedy",
        "LinearICF": "linear_icf",
        "LinearThompsonSampling": "linear_thompson_sampling",
        "LinearUCB": "linear_ucb",
        "LinearUCB1": "linear_ucb1",
        "LogPopEnt": "log_pop_ent",
        "MFValueFunction": "mf_value_function",
        "MostPopular": "most_popular",
        "NICF": "nicf",
        "PTS": "pts",
        "Random": "random",
        "ThompsonSampling": "thompson_sampling",
        "UCB": "ucb",
        "ValueFunction": "value_function",
    }

    @classmethod
    def all(cls: VFRegistry) -> List[str]:
        return list(cls._vf.keys())

    @classmethod
    def get(cls: VFRegistry, name: str):
        return _import_class(
            root="irec.agents.value_functions",
            module_name=cls._vf[name],
            class_name=name
        )


class ASPRegistry:
    
    from irec.agents.action_selection_policies.base import ActionSelectionPolicy
    from irec.agents.action_selection_policies.egreedy import ASPEGreedy
    from irec.agents.action_selection_policies.generic_greedy import ASPGenericGreedy
    from irec.agents.action_selection_policies.greedy import ASPGreedy
    from irec.agents.action_selection_policies.ic_greedy import ASPICGreedy
    from irec.agents.action_selection_policies.reranker import ASPReranker

    _asp = {
        "ActionSelectionPolicy": ActionSelectionPolicy,
        "ASPEGreedy": ASPEGreedy,
        "ASPGenericGreedy": ASPGenericGreedy,
        "ASPGreedy": ASPGreedy,
        "ASPICGreedy": ASPICGreedy,
        "ASPReranker": ASPReranker,
    }

    @classmethod
    def all(cls: AgentRegistry) -> List[str]:
        return list(cls._asp.keys())

    @classmethod
    def get(cls: AgentRegistry, name: str):
        return cls._asp[name]