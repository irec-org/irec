from irec.metrics import ILD, Recall, Precision, EPC, EPD
from irec.agents.value_functions.Entropy import Entropy
from .MetricEvaluator import MetricEvaluator
from collections import defaultdict
from irec.environment.dataset import Dataset
import irec.agents.value_functions
import scipy.sparse
import numpy as np
import time
from irec import metrics
np.seterr(all="raise")

class InteractionMetricEvaluator(MetricEvaluator):
    def __init__(
        self,
        ground_truth_dataset,
        num_interactions,
        interaction_size,
        interactions_to_evaluate=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ground_truth_dataset = ground_truth_dataset
        self.num_interactions = num_interactions
        self.interaction_size = interaction_size
        self.interactions_to_evaluate = interactions_to_evaluate
        if self.interactions_to_evaluate == None:
            self.interactions_to_evaluate = list(range(self.num_interactions))
        self.iterations_to_evaluate = self.interactions_to_evaluate

        # print(">>>>>>>>>>>", self.ground_truth_dataset.num_total_users)
        # print(">>>>>>>>>>>", self.ground_truth_dataset.num_total_items)

        if isinstance(ground_truth_dataset, Dataset):
            self.ground_truth_consumption_matrix = scipy.sparse.csr_matrix(
                (
                    self.ground_truth_dataset.data[:, 2],
                    (
                        self.ground_truth_dataset.data[:, 0],
                        self.ground_truth_dataset.data[:, 1],
                    ),
                ),
                # (
                # np.max(self.ground_truth_dataset.uids),
                # np.max(self.ground_truth_dataset.iids),
                # self.ground_truth_dataset.num_total_items,
                # ),
                (
                    self.ground_truth_dataset.num_total_users,
                    self.ground_truth_dataset.num_total_items,
                ),
            )

    def _metric_evaluation(self, metric_class):
        start_time = time.time()
        metric_values = []
        for i in range(self.num_interactions):
            if issubclass(metric_class, Recall):
                metric = metric_class(
                    users_false_negative=self.users_false_negative,
                    ground_truth_dataset=self.ground_truth_dataset,
                    relevance_evaluator=self.relevance_evaluator,
                )
            elif issubclass(metric_class, ILD):
                metric = metric_class(
                    items_distance=self.items_distance,
                    ground_truth_dataset=self.ground_truth_dataset,
                    relevance_evaluator=self.relevance_evaluator,
                )
            elif issubclass(metric_class, EPC):
                metric = metric_class(
                    items_normalized_popularity=self.items_normalized_popularity,
                    ground_truth_dataset=self.ground_truth_dataset,
                    relevance_evaluator=self.relevance_evaluator,
                )
            else:
                metric = metric_class(
                    ground_truth_dataset=self.ground_truth_dataset,
                    relevance_evaluator=self.relevance_evaluator,
                )
            for uid in self.uids:
                interaction_results = self.users_items_recommended[uid][
                    i * self.interaction_size : i * self.interaction_size
                    + self.interaction_size
                ]
                for item in interaction_results:
                    metric.update_recommendation(
                        uid, item, self.ground_truth_consumption_matrix[uid, item]
                    )

            metric_values.append(np.mean([metric.compute(uid) for uid in self.uids]))

        print(
            f"{self.__class__.__name__} spent {time.time()-start_time:.2f} seconds executing {metric_class.__name__} metric"
        )
        return metric_values

    def evaluate(
        self,
        metric_class,
        results,
    ):
        self.users_false_negative = defaultdict(int)
        for row in self.ground_truth_dataset.data:
            uid = int(row[0])
            reward = row[2]
            if self.relevance_evaluator.is_relevant(reward):
                self.users_false_negative[uid] += 1

        self.users_items_recommended = defaultdict(list)
        for uid, item in results:
            self.users_items_recommended[uid].append(item)
        self.uids = list(self.users_items_recommended.keys())
        if issubclass(metric_class, (ILD,EPD)):
            self.items_distance = metrics.get_items_distance(
                self.ground_truth_consumption_matrix
            )
        if issubclass(metric_class, EPC):
            self.items_normalized_popularity = (
                irec.agents.value_functions.MostPopular.MostPopular.get_items_popularity(
                    self.ground_truth_consumption_matrix
                )
            )
        if issubclass(metric_class, Entropy):
            self.items_entropy = irec.agents.value_functions.Entropy.Entropy.get_items_entropy(
                self.ground_truth_consumption_matrix
            )

        metric_values = self._metric_evaluation(metric_class)

        return metric_values

