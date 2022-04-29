from irec.offline_experiments.relevance_evaluator import ThresholdRelevanceEvaluator
import numpy as np

np.seterr(all="raise")


class MetricEvaluator:
    """MetricsEvaluator."""

    def __init__(self, relevance_evaluator_threshold: float, *args, **kwargs):
        """__init__.

        Args:
            relevance_evaluator_threshold (float): relevance_evaluator_threshold
            args:
            kwargs:
        """
        del args, kwargs
        self.relevance_evaluator = ThresholdRelevanceEvaluator(
            relevance_evaluator_threshold
        )
