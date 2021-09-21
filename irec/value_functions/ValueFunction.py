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

from typing import Any


class ValueFunction:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, observation: Any):
        train_dataset = observation
        pass
        # super().reset(train_dataset)

    def action_estimates(self, candidate_actions: CandidateActions):
        # uid = candidate_actions[0]
        # candidate_items = candidate_actions[1]
        return None, None

    def update(self, observation, action: CandidateAction, reward: int, info: dict):
        # uid = action[0]
        # item = action[1]
        # additional_data = info
        pass
