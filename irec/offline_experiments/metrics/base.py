from irec.offline_experiments.relevance_evaluator import RelevanceEvaluator
import numpy as np
from typing import Any

np.seterr(all="raise")

"""Evaluation Metrics.

This module implements numerous evaluation metrics widely used in RS.
"""

class Metric:
    """Metric.

    Metrics are used to assess the performance of a recommendation system.
    For this, there are several metrics capable of evaluating recommendations in different ways.
    """

    def __init__(
        self,
        ground_truth_dataset: Any,
        relevance_evaluator: RelevanceEvaluator,
    ):
        """__init__.

        Args:
            ground_truth_dataset (Any): ground_truth_dataset
            relevance_evaluator (RelevanceEvaluator): relevance_evaluator
            args:
            kwargs:
        """

        self.ground_truth_dataset = ground_truth_dataset
        self.relevance_evaluator = relevance_evaluator

    def compute(self, uid: int) -> Any:
        """compute.

        This method performs the metric calculation for a given user.

        Args:
            uid (int): uid

        Returns:
            Any:
        """

        return None

    def update_recommendation(self, uid: int, item: int, reward: float) -> None:
        """update_recommendation.

        Uses user-supplied item rating to update metric attributes.

        Args:
            uid (int): uid
            item (int): item
            reward (float): reward

        Returns:
            None:
        """
        raise NotImplementedError

    def update_consumption_history(self, uid: int, item: int, reward: float) -> None:
        """update_consumption_history.

        Update items consumed by a user.

        Args:
            uid (int): uid
            item (int): item
            reward (float): reward

        Returns:
            None:
        """

        raise NotImplementedError