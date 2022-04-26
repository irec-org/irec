from .MetricEvaluator import MetricEvaluator
from collections import defaultdict
from irec.offline_experiments.metrics.recall import Recall
from irec.utils import dataset
import scipy.sparse
import numpy as np
import time
np.seterr(all="raise")

class TotalMetricEvaluator(MetricEvaluator):
    def __init__(self, ground_truth_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ground_truth_dataset = ground_truth_dataset
        if isinstance(ground_truth_dataset, dataset.Dataset):
            self.ground_truth_consumption_matrix = scipy.sparse.csr_matrix(
                (
                    self.ground_truth_dataset.data[:, 2],
                    (
                        self.ground_truth_dataset.data[:, 0],
                        self.ground_truth_dataset.data[:, 1],
                    ),
                ),
                (
                    self.ground_truth_dataset.num_total_users,
                    self.ground_truth_dataset.num_total_items,
                ),
            )

    def _metric_evaluation(self, metric_class):
        start_time = time.time()
        metric_values = []
        if issubclass(metric_class, Recall):
            metric = metric_class(
                users_false_negative=self.users_false_negative,
                ground_truth_dataset=self.ground_truth_dataset,
                relevance_evaluator=self.relevance_evaluator,
            )
        else:
            metric = metric_class(
                ground_truth_dataset=self.ground_truth_dataset,
                relevance_evaluator=self.relevance_evaluator,
            )
        start_time = time.time()
        i = 0
        while i < len(self.results):
            uid = self.results[i][0]
            item = self.results[i][1]
            metric.update_recommendation(
                uid, item, self.ground_truth_consumption_matrix[uid, item]
            )
            i += 1

        metric_values.append([metric.compute(uid) for uid in self.uids])

        print(
            f"{self.__class__.__name__} spent {time.time()-start_time:.2f} seconds executing {metric_class.__name__} metric"
        )
        return metric_values

    def evaluate(self, metric_class, results):
        self.users_false_negative = defaultdict(int)
        for row in self.ground_truth_dataset.data:
            uid = int(row[0])
            reward = row[2]
            if self.relevance_evaluator.is_relevant(reward):
                self.users_false_negative[uid] += 1
        uids = []
        for uid, _ in results:
            uids.append(uid)
        uids = list(set(uids))
        self.uids = uids
        self.results = results

        metric_values = self._metric_evaluation(metric_class)

        return metric_values

