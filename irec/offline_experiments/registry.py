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



class MetricRegistry:

    from irec.offline_experiments.metrics.base import Metric
    from irec.offline_experiments.metrics.ild import ILD
    from irec.offline_experiments.metrics.recall import Recall
    from irec.offline_experiments.metrics.precision import Precision
    from irec.offline_experiments.metrics.epc import EPC
    from irec.offline_experiments.metrics.epd import EPD
    from irec.offline_experiments.metrics.gini_coefficient_inv import GiniCoefficientInv
    from irec.offline_experiments.metrics.users_coverage import UsersCoverage
    from irec.offline_experiments.metrics.hits import Hits
    
    _metric = {
        "Metric": Metric,
        "ILD": ILD,
        "Recall": Recall,
        "Precision": Precision,
        "EPC": EPC,
        "EPD": EPD,
        "GiniCoefficientInv": GiniCoefficientInv,
        "UsersCoverage": UsersCoverage,
        "Hits": Hits,
    }

    @classmethod
    def all(cls: MetricRegistry) -> List[str]:
        return list(cls._metric.keys())

    @classmethod
    def get(cls: MetricRegistry, name: str):
        return cls._metric[name]


