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