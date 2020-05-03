from .Saveable import Saveable
import numpy as np
from collections import defaultdict
import util.metrics as metrics

class MetricsEvaluator(Saveable):
    METRICS_PRETTY = {'precision':'Precision','hits':'Hits','cumulative_precision':'Cumulative Precision',
                      'recall':'Recall','f1':'F1 Score','ndcg':'NDCG'}
    def __init__(self, name, k):
        super().__init__()
        self.metrics = defaultdict(dict)
        self.metrics_mean = defaultdict(float)
        self.name = name
        self.k = k

    def eval_chunk_metrics(self, result, ground_truth, size):
        self.metrics.clear()
        self.metrics_mean.clear()
        lowest_value = np.min(ground_truth)
        threshold = np.unique(ground_truth)[-2]
        for uid, predicted in result.items():
            predicted = predicted[self.k-size:self.k]
            actual = np.nonzero(ground_truth[uid,:]>=threshold)[0]
            hits = len(set(predicted) & set(actual))
            precision = hits/size
            self.metrics_mean['precision'] += precision
            self.metrics_mean['hits'] += hits
        for k in self.metrics_mean.keys():
            self.metrics_mean[k] /= len(result)
        self.save()
        pass
    def eval_metrics(self, result, ground_truth):
        self.metrics.clear()
        self.metrics_mean.clear()
        lowest_value = np.min(ground_truth)
        threshold = np.unique(ground_truth)[-2]
        for uid, predicted in result.items():
            predicted = predicted[:self.k]
            actual = np.nonzero(ground_truth[uid,:]>=threshold)[0]
            hits = len(set(predicted) & set(actual))
            precision = hits/len(predicted)
            recall = hits/len(actual)
            self.metrics_mean['precision'] += precision
            self.metrics_mean['recall'] += recall
            self.metrics_mean['hits'] += hits
        for k in self.metrics_mean.keys():
            self.metrics_mean[k] /= len(result)
        self.save()
