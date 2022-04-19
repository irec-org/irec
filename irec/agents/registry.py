from __future__ import annotations
from typing import List

def _import_class(
        root:str,
        module_name:str,
        class_name:str
    ):

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
        "BestRated": "experimental.best_rated",
        "cluster_bandit": "experimental.cb",
        "COFIBA": "matrix_factorization.cofiba",
        "EGreedy": "experimental.e_greedy",
        "Entropy": "experimental.entropy",
        "Entropy0": "experimental.entropy0",
        "ExperimentalValueFunction": "experimental.experimental_valueFunction",
        "GenericThompsonSampling": "experimental.generic_thompson_sampling",
        "GLM_UCB": "matrix_factorization.glm_ucb",
        "HELF": "experimental.helf",
        "ICF": "matrix_factorization.icf",
        "ICTRTS": "matrix_factorization.ictr",
        "kNNBandit": "experimental.knn_bandit",
        "LinEGreedy": "matrix_factorization.lin_egreedy",
        "LinUCB": "matrix_factorization.lin_ucb",
        "LinearEGreedy": "matrix_factorization.linear_egreedy",
        "LinearICF": "matrix_factorization.linear_icf",
        "LinearThompsonSampling": "matrix_factorization.linear_ts",
        "LinearUCB": "matrix_factorization.linear_ucb",
        "LinearUCB1": "matrix_factorization.linear_ucb1",
        "LogPopEnt": "experimental.log_pop_ent",
        "MFValueFunction": "matrix_factorization.mf_value_function",
        "MostPopular": "experimental.most_popular",
        "NICF": "experimental.nicf",
        "PTS": "matrix_factorization.pts",
        "Random": "experimental.random",
        "ThompsonSampling": "experimental.thompson_sampling",
        "UCB": "experimental.ucb",
        "ValueFunction": "base",
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
    def all(cls: ASPRegistry) -> List[str]:
        return list(cls._asp.keys())

    @classmethod
    def get(cls: ASPRegistry, name: str):
        return _import_class(
            root="irec.agents.action_selection_policies",
            module_name=cls._asp[name],
            class_name=name
        )