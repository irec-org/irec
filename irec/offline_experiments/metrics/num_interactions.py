from collections import defaultdict
from irec.offline_experiments.metrics.base import Metric


class NumInteractions(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_num_interactions = defaultdict(int)

    def compute(self, uid):
        return self.users_num_interactions[uid]

    def update_recommendation(self, uid, item, reward):
        self.users_num_interactions[uid] += 1

