from .InteractionMetricEvaluator import InteractionMetricEvaluator
from irec.offline_experiments.metrics import ILD, Recall, Precision, EPC, EPD
import numpy as np
import time
from typing import Any

np.seterr(all="raise")


class IterationsMetricEvaluator(InteractionMetricEvaluator):
    @staticmethod
    def metric_summarize(users_metric_values):
        return np.mean(users_metric_values)

    def _metric_evaluation(self, metric_class):
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
        if 0 not in self.iterations_to_evaluate:
            self.iterations_to_evaluate: Any = [0] + self.iterations_to_evaluate
        for i in range(len(self.iterations_to_evaluate) - 1):
            for uid in self.uids:
                interaction_results = self.users_items_recommended[uid][
                    self.iterations_to_evaluate[i] : self.iterations_to_evaluate[i + 1]
                ]
                for item in interaction_results:
                    metric.update_recommendation(
                        uid, item, self.ground_truth_consumption_matrix[uid, item]
                    )
            # if (i+1) in interactions_to_evaluate:
            print(
                f"Computing iteration {self.iterations_to_evaluate[i+1]} with {self.__class__.__name__}"
            )
            metric_values.append(
                self.metric_summarize([metric.compute(uid) for uid in self.uids])
            )

        print(
            f"{self.__class__.__name__} spent {time.time()-start_time:.2f} seconds executing {metric_class.__name__} metric"
        )
        return metric_values
