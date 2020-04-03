from .Saveable import Saveable
import numpy as np
from collections import defaultdict

class MetricsEvaluator(Saveable):
    def __init__(self, name, k# ,threshold=4
    ):
        super().__init__()
        self.metrics = defaultdict(dict)
        self.metrics_mean = defaultdict(float)
        self.name = name
        self.k = k
        # self.threshold = threshold

    def eval_metrics(self, result, ground_truth):
        self.metrics.clear()
        self.metrics_mean.clear()
        for uid, predicted in result.items():
            predicted = predicted[:self.k]
            actual = np.nonzero(ground_truth[uid,:]# >=self.threshold
            )[0]
            precision = self.precision(predicted, actual)
            recall = self.recall(predicted, actual)
            self.metrics_mean['precision'] += precision
            self.metrics_mean['recall'] += recall
            self.metrics[uid] = {'precision': precision,
                                 'recall': recall}
        for k in self.metrics_mean.keys():
            self.metrics_mean[k] /= len(result)
        self.save()

    def precision(self, predicted, actual):
        return len(set(predicted) & set(actual))/len(predicted)
    def recall(self, predicted, actual):
        return len(set(predicted) & set(actual))/len(actual)
