from collections import defaultdict
from irec.offline_experiments.metrics.base import Metric


class AP(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_true_positive = defaultdict(int)
        self.users_false_positive = defaultdict(int)
        self.users_cumulated_precision = defaultdict(float)
        self.users_num_recommendations = defaultdict(int)

    def compute(self, uid):
        return self.users_cumulated_precision[uid] / self.users_num_recommendations

    def update_recommendation(self, uid, item, reward):
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid] += 1
        else:
            self.users_false_positive[uid] += 1

        self.users_cumulated_precision[uid] += self.users_true_positive[uid] / (
            self.users_true_positive[uid] + self.users_false_positive[uid]
        )
        self.users_num_recommendations[uid] += 1