from __future__ import annotations
from typing import List

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
    
    from irec.agents.value_functions.best_rated import BestRated
    from irec.agents.value_functions.cb import CB
    from irec.agents.value_functions.cofiba import COFIBA
    from irec.agents.value_functions.distinct_popular import DistinctPopular
    from irec.agents.value_functions.e_greedy import EGreedy
    from irec.agents.value_functions.EMostPopular import EMostPopular
    from irec.agents.value_functions.entropy import Entropy
    from irec.agents.value_functions.entropy0 import Entropy0
    from irec.agents.value_functions.experimental_valueFunction import ExperimentalValueFunction
    from irec.agents.value_functions.generic_thompson_sampling import GenericThompsonSampling
    from irec.agents.value_functions.glm_ucb import GLM_UCB
    from irec.agents.value_functions.helf import HELF
    from irec.agents.value_functions.ic_bandit import ICLinUCB
    from irec.agents.value_functions.icf import ICF
    from irec.agents.value_functions.ictr import ICTRTS
    from irec.agents.value_functions.knn_bandit import kNNBandit
    from irec.agents.value_functions.lin_egreedy import LinEGreedy
    from irec.agents.value_functions.lin_ucb_pca import LinUCBPCA
    from irec.agents.value_functions.lin_ucb_var import LinUCBvar
    from irec.agents.value_functions.lin_ucb import LinUCB
    from irec.agents.value_functions.linear_egreedy_init import LinearEGreedyInit
    from irec.agents.value_functions.linear_egreedy import LinearEGreedy
    from irec.agents.value_functions.linear_icf import LinearICF
    from irec.agents.value_functions.linear_thompson_sampling import LinearThompsonSampling
    from irec.agents.value_functions.linear_ucb_init import LinearUCBInit
    from irec.agents.value_functions.linear_ucb import LinearUCB
    from irec.agents.value_functions.linear_ucb1 import LinearUCB1
    from irec.agents.value_functions.log_pop_ent import LogPopEnt
    from irec.agents.value_functions.mf_value_function import MFValueFunction
    from irec.agents.value_functions.most_popular import MostPopular
    from irec.agents.value_functions.most_representative import MostRepresentative
    from irec.agents.value_functions.nicf import NICF
    from irec.agents.value_functions.pop_plus_ent import PopPlusEnt
    from irec.agents.value_functions.ppelpe import PPELPE
    from irec.agents.value_functions.pts import PTS
    from irec.agents.value_functions.random import Random
    from irec.agents.value_functions.thompson_sampling import ThompsonSampling
    from irec.agents.value_functions.ucb_learner import UCBLearner
    from irec.agents.value_functions.ucb import UCB
    from irec.agents.value_functions.value_function import ValueFunction
    from irec.agents.value_functions.wspb_init import WSPBInit
    from irec.agents.value_functions.wspb_pca import WSPBPCA
    from irec.agents.value_functions.wspb_var import WSPBvar
    from irec.agents.value_functions.wspb import WSPB

    _vf = {
        "BestRated": BestRated,
        "CB": CB,
        "COFIBA": COFIBA,
        "DistinctPopular": DistinctPopular,
        "EGreedy": EGreedy,
        "EMostPopular": EMostPopular,
        "Entropy": Entropy,
        "Entropy0": Entropy0,
        "ExperimentalValueFunction": ExperimentalValueFunction,
        "GenericThompsonSampling": GenericThompsonSampling,
        "GLM_UCB": GLM_UCB,
        "HELF": HELF,
        "ICLinUCB": ICLinUCB,
        "ICF": ICF,
        "ICTRTS": ICTRTS,
        "kNNBandit": kNNBandit,
        "LinEGreedy": LinEGreedy,
        "LinUCBPCA": LinUCBPCA,
        "LinUCBvar": LinUCBvar,
        "LinUCB": LinUCB,
        "LinearEGreedyInit": LinearEGreedyInit,
        "LinearEGreedy": LinearEGreedy,
        "LinearICF": LinearICF,
        "LinearThompsonSampling": LinearThompsonSampling,
        "LinearUCBInit": LinearUCBInit,
        "LinearUCB": LinearUCB,
        "LinearUCB1": LinearUCB1,
        "LogPopEnt": LogPopEnt,
        "MFValueFunction": MFValueFunction,
        "MostPopular": MostPopular,
        "MostRepresentative": MostRepresentative,
        "NICF": NICF,
        "PopPlusEnt": PopPlusEnt,
        "PPELPE": PPELPE,
        "PTS": PTS,
        "Random": Random,
        "ThompsonSampling": ThompsonSampling,
        "UCBLearner": UCBLearner,
        "UCB": UCB,
        "ValueFunction": ValueFunction,
        "WSPBInit": WSPBInit,
        "WSPBPCA": WSPBPCA,
        "WSPBvar": WSPBvar,
        "WSPB": WSPB
    }

    @classmethod
    def all(cls: VFRegistry) -> List[str]:
        return list(cls._vf.keys())

    @classmethod
    def get(cls: VFRegistry, name: str):
        return cls._vf[name]


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