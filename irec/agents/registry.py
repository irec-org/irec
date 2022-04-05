from __future__ import annotations
from typing import List

def _import_class(root:str,
              module_name:str,
              class_name:str):

    exec(f"from {root}.{module_name} import {class_name}")

    return eval(class_name)

class AgentRegistry:
    
    _agent = {
        "Agent": "base",
        "SimpleAgent": "simple_agent",
        "SimpleEnsembleAgent": "simple_ensemble_agent",
    }

    @classmethod
    def all(cls: AgentRegistry) -> List[str]:
        return list(cls._agent.keys())

    @classmethod
    def get(cls: AgentRegistry, name: str):
         return _import_class(
            root="irec.agents",
            module_name=cls._agent[name],
            class_name=name
        )


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
    
    _asp = {
        "ActionSelectionPolicy": "base",
        "ASPEGreedy": "egreedy",
        "ASPGenericGreedy": "generic_greedy",
        "ASPGreedy": "greedy",
        "ASPICGreedy": "ic_greedy",
        "ASPReranker": "reranker",
    }


    @classmethod
    def all(cls: AgentRegistry) -> List[str]:
        return list(cls._asp.keys())

    @classmethod
    def get(cls: AgentRegistry, name: str):
        return _import_class(
            root="irec.agents.action_selection_policies",
            module_name=cls._asp[name],
            class_name=name
        )