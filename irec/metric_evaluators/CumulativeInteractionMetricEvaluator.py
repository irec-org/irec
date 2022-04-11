from .InteractionMetricEvaluator import InteractionMetricEvaluator
from irec.metrics import ILD, Recall, Precision, EPC, EPD
from irec.agents.value_functions.experimental.entropy import Entropy
import numpy as np
import time
np.seterr(all="raise")

class CumulativeInteractionMetricEvaluator(InteractionMetricEvaluator):
    @staticmethod
    def metric_summarize(users_metric_values):
        return np.mean(list(users_metric_values.values()))

    def _metric_evaluation(self, metric_class):
        start_time = time.time()
        metric_values = []
        if issubclass(metric_class, Recall):
            metric = metric_class(
                users_false_negative=self.users_false_negative,
                ground_truth_dataset=self.ground_truth_dataset,
                relevance_evaluator=self.relevance_evaluator,
            )
        elif issubclass(metric_class, (ILD,EPD)):
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
        for i in range(self.num_interactions):
            for uid in self.uids:
                interaction_results = self.users_items_recommended[uid][
                    i * self.interaction_size : i * self.interaction_size
                    + self.interaction_size
                ]
                for item in interaction_results:
                    metric.update_recommendation(
                        uid, item, self.ground_truth_consumption_matrix[uid, item]
                    )

            if (i + 1) in self.interactions_to_evaluate:
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
