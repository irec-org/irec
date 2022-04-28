from __future__ import annotations
from typing import List

class MetricEvaluatorRegistry:

    from irec.offline_experiments.metric_evaluators.base import MetricEvaluator
    from irec.offline_experiments.metric_evaluators.cumulative_interaction import CumulativeInteraction
    from irec.offline_experiments.metric_evaluators.cumulative import Cumulative
    from irec.offline_experiments.metric_evaluators.interaction import Interaction
    from irec.offline_experiments.metric_evaluators.iterations import Interaction as Iterations
    from irec.offline_experiments.metric_evaluators.stage_iterations import StageIterations
    from irec.offline_experiments.metric_evaluators.total import Total
    from irec.offline_experiments.metric_evaluators.user_cumulative_interaction import UserCumulativeInteraction
    
    _metric_eval = {
        "MetricEvaluator": MetricEvaluator,
        "CumulativeInteraction": CumulativeInteraction,
        "Cumulative": Cumulative,
        "Interaction": Interaction,
        "Iterations": Iterations,
        "StageIterations": StageIterations,
        "Total": Total,
        "UserCumulativeInteraction": UserCumulativeInteraction,
    }

    @classmethod
    def all(cls: MetricEvaluatorRegistry) -> List[str]:
        return list(cls._metric_eval.keys())

    @classmethod
    def get(cls: MetricEvaluatorRegistry, name: str):
        return cls._metric_eval[name]


