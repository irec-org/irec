from irec.environment.dataset import Dataset
from irec.agents import Agent

"""Evaluation Policies.

This module implements several assessment policies that will define how 
to assess the agent after the recommendation process.
"""

class EvaluationPolicy:
    """EvaluationPolicy.
        
    Defines a form of evaluation for the recommendation process.
    """

    def evaluate(
        self, model: Agent, train_dataset: Dataset, test_dataset: Dataset
    ) -> [list, dict]:
        """evaluate.
        
        Performs the form of evaluation according to the chosen policy.

        Args:
            model (Agent): model
            train_dataset (Dataset): train_dataset
            test_dataset (Dataset): test_dataset

        Returns:
            Tuple[list, dict]:
        """
        pass