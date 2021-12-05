class RelevanceEvaluator:
    """RelevanceEvaluator."""

    def __init__(self, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
        """
        del args, kwargs
        pass
        # super().__init__(*args, **kwargs)

    def is_relevant(self, reward: float):
        """is_relevant.

        Args:
            reward (float): reward
        """
        raise NotImplementedError


class ThresholdRelevanceEvaluator(RelevanceEvaluator):
    """ThresholdRelevanceEvaluator."""

    def __init__(self, threshold: float, *args, **kwargs):
        """__init__.

        Args:
            threshold (float): threshold
            args:
            kwargs:
        """
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def is_relevant(self, reward: float):
        """is_relevant.

        Args:
            reward (float): reward
        """
        return reward > self.threshold
