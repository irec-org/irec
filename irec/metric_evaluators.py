import numpy as np

import irec.value_functions

import scipy.sparse
from collections import defaultdict
import time
from irec.utils.utils import run_parallel
import ctypes

# import irec.utils.dataset as dataset
from irec.utils import dataset
from irec.value_functions.Entropy import Entropy
from irec.RelevanceEvaluator import ThresholdRelevanceEvaluator

np.seterr(all="raise")


class MetricsEvaluator:
    """MetricsEvaluator."""

    def __init__(
        self, relevance_evaluator=ThresholdRelevanceEvaluator(3.999), *args, **kwargs
    ):
        """__init__.

        Args:
            relevance_evaluator:
            args:
            kwargs:
        """
        super().__init__(*args, **kwargs)
        self.relevance_evaluator = relevance_evaluator


class TotalMetricsEvaluator(MetricsEvaluator):
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


class CumulativeMetricsEvaluator(MetricsEvaluator):
    def __init__(self, buffer_size, ground_truth_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ground_truth_dataset = ground_truth_dataset
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
        self.buffer_size = buffer_size

    def _metric_evaluation(self, metric_class):
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

        start = 0
        start_time = time.time()
        metric_values = []
        while start < len(self.results):
            for i in range(start, min(start + self.buffer_size, len(self.results))):
                uid = self.results[i][0]
                item = self.results[i][1]
                metric.update_recommendation(
                    uid, item, self.ground_truth_consumption_matrix[uid, item]
                )
            start = min(start + self.buffer_size, len(self.results))
            metric_values.append(np.mean([metric.compute(uid) for uid in self.uids]))
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


class InteractionMetricsEvaluator(MetricsEvaluator):
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

    def _metric_evaluation(
        self, metric_class, num_interactions, interaction_size, interactions_to_evaluate
    ):
        start_time = time.time()
        metric_values = []
        for i in range(num_interactions):
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
                    i * interaction_size : i * interaction_size + interaction_size
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
        num_interactions,
        interaction_size,
        results,
        interactions_to_evaluate=None,
    ):
        if interactions_to_evaluate == None:
            interactions_to_evaluate = list(range(num_interactions))
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
        if isinstance(metric_class, ILD):
            self.items_distance = get_items_distance(
                self.ground_truth_consumption_matrix
            )
        if isinstance(metric_class, EPC):
            self.items_normalized_popularity = (
                irec.value_functions.MostPopular.get_items_popularity(
                    self.ground_truth_consumption_matrix
                )
            )
        if isinstance(metric_class, Entropy):
            self.items_entropy = irec.value_functions.Entropy.get_items_entropy(
                self.ground_truth_consumption_matrix
            )

        metric_values = self._metric_evaluation(
            metric_class, num_interactions, interaction_size, interactions_to_evaluate
        )

        return metric_values


class CumulativeInteractionMetricsEvaluator(InteractionMetricsEvaluator):
    @staticmethod
    def metric_summarize(users_metric_values):
        return np.mean(list(users_metric_values.values()))

    def _metric_evaluation(
        self, metric_class, num_interactions, interaction_size, interactions_to_evaluate
    ):
        start_time = time.time()
        metric_values = []
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
        elif issubclass(metric_class, Entropy):
            metric = metric_class(
                items_entropy=self.items_entropy,
                ground_truth_dataset=self.ground_truth_dataset,
                relevance_evaluator=self.relevance_evaluator,
            )
        else:
            metric = metric_class(
                ground_truth_dataset=self.ground_truth_dataset,
                relevance_evaluator=self.relevance_evaluator,
            )
        for i in range(num_interactions):
            for uid in self.uids:
                interaction_results = self.users_items_recommended[uid][
                    i * interaction_size : i * interaction_size + interaction_size
                ]
                for item in interaction_results:
                    metric.update_recommendation(
                        uid, item, self.ground_truth_consumption_matrix[uid, item]
                    )

            if (i + 1) in interactions_to_evaluate:
                print(f"Computing interaction {i+1} with {self.__class__.__name__}")
                metric_values.append(
                    self.metric_summarize(
                        {uid: metric.compute(uid) for uid in self.uids}
                    )
                )

        print(
            f"{self.__class__.__name__} spent {time.time()-start_time:.2f} seconds executing {metric_class.__name__} metric"
        )
        return metric_values


class UserCumulativeInteractionMetricsEvaluator(CumulativeInteractionMetricsEvaluator):
    @staticmethod
    def metric_summarize(users_metric_values):
        return users_metric_values


class IterationsMetricsEvaluator(InteractionMetricsEvaluator):
    @staticmethod
    def metric_summarize(users_metric_values):
        return np.mean(users_metric_values)

    def _metric_evaluation(
        self, metric_class, num_interactions, interaction_size, iterations_to_evaluate
    ):
        start_time = time.time()
        metric_values = []
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
        if 0 not in iterations_to_evaluate:
            iterations_to_evaluate = [0] + iterations_to_evaluate
        for i in range(len(iterations_to_evaluate) - 1):
            for uid in self.uids:
                interaction_results = self.users_items_recommended[uid][
                    iterations_to_evaluate[i] : iterations_to_evaluate[i + 1]
                ]
                for item in interaction_results:
                    metric.update_recommendation(
                        uid, item, self.ground_truth_consumption_matrix[uid, item]
                    )
            # if (i+1) in interactions_to_evaluate:
            print(
                f"Computing iteration {iterations_to_evaluate[i+1]} with {self.__class__.__name__}"
            )
            metric_values.append(
                self.metric_summarize([metric.compute(uid) for uid in self.uids])
            )

        print(
            f"{self.__class__.__name__} spent {time.time()-start_time:.2f} seconds executing {metric_class.__name__} metric"
        )
        return metric_values
