from collections import defaultdict
from irec.offline_experiments.metrics.base import Metric


class Entropy(Metric):
    def __init__(self, items_entropy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_num_items_recommended = defaultdict(int)
        self.users_entropy_cumulated = defaultdict(float)
        self.items_entropy = items_entropy

    def compute(self, uid):
        return self.users_entropy_cumulated[uid] / self.users_num_items_recommended[uid]

    def update_recommendation(self, uid, item, reward):
        self.users_num_items_recommended[uid] += 1
        self.users_entropy_cumulated[uid] += self.items_entropy[item]
