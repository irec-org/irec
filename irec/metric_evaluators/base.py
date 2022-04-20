from irec.RelevanceEvaluator import ThresholdRelevanceEvaluator
import numpy as np

np.seterr(all="raise")


class MetricEvaluator:
    """MetricsEvaluator.
    
    This module aims to guide the entire evaluation process
    over the logs from each iteration of the Evaluation Policy. 
    As the iRec stores each execution log, the researcher can define
    how s/he would like to evaluate the actions selected by the
    recommendation model after all interactions.
    
    """

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
