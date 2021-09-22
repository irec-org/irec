from os.path import dirname, realpath, sep, pardir
import sys, os

# sys.path.append(dirname(dirname(realpath(__file__))))
from irec.CandidateActions import CandidateActions
from irec.CandidateAction import CandidateAction

import numpy as np
import scipy.sparse

from collections import defaultdict
import pickle
import json

from typing import Any, Tuple


class ValueFunction:
    """ValueFunction."""

    def __init__(self, *args, **kwargs):
        """__init__.

        Args:
            args:
            kwargs:
        """
        super().__init__(*args, **kwargs)

    def reset(self, observation: Any):
        """reset.

        Args:
            observation (Any): observation
        """
        train_dataset = observation
        pass
        # super().reset(train_dataset)

    def action_estimates(self, candidate_actions: CandidateActions) -> Tuple[Any, dict]:
        """action_estimates.

        Args:
            candidate_actions (CandidateActions): candidate_actions

        Returns:
            Tuple[Any, dict]: Scores and information
        """

        # uid = candidate_actions[0]
        # candidate_items = candidate_actions[1]
        return None, None

    def update(
        self, observation: Any, action: CandidateAction, reward: float, info: dict
    ) -> None:
        """update.

        Args:
            observation (Any): observation
            action (CandidateAction): action
            reward (float): reward
            info (dict): info
        """
        # uid = action[0]
        # item = action[1]
        # additional_data = info
        return None
