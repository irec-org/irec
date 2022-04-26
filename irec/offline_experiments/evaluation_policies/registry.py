from __future__ import annotations
from typing import List


class EvalPolicyRegistry:

    from irec.offline_experiments.evaluation_policies.base import EvaluationPolicy
    from irec.offline_experiments.evaluation_policies.fixed_interaction import FixedInteraction
    from irec.offline_experiments.evaluation_policies.limited_interaction import LimitedInteraction

    _eval_policy = {
        "EvaluationPolicy": EvaluationPolicy,
        "FixedInteraction": FixedInteraction,
        "LimitedInteraction": LimitedInteraction,
    }

    @classmethod
    def all(cls: EvalPolicyRegistry) -> List[str]:
        return list(cls._eval_policy.keys())

    @classmethod
    def get(cls: EvalPolicyRegistry, name: str):
        return cls._eval_policy[name]


