from collections import defaultdict
import numpy as np
from irec.offline_experiments.metrics.base import Metric


class TopItemsMembership(Metric):
    def __init__(self, items_feature_values, top_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_num_items_recommended = defaultdict(int)
        self.users_membership_count_cumulated = defaultdict(float)
        self.items_feature_values = items_feature_values
        self.top_size = top_size
        self.top_items = set(np.argsort(items_feature_values)[::-1][: self.top_size])

    def compute(self, uid):
        return (
            self.users_membership_count_cumulated[uid]
            / self.users_num_items_recommended[uid]
        )

    def update_recommendation(self, uid, item, reward):
        self.users_num_items_recommended[uid] += 1
        if item in self.top_items:
            self.users_membership_count_cumulated[uid] += 1
