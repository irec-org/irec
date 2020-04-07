from .Saveable import Saveable
import numpy as np
from collections import defaultdict

class MetricsEvaluator(Saveable):
    METRICS_PRETTY = {'precision':'Precision','hits':'Hits'}
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
        
        for uid, predicted in result.items():
            predicted = predicted[self.k-size:self.k]
            actual = np.nonzero(ground_truth[uid,:]>lowest_value)[0]
            hits = len(set(predicted) & set(actual))
            precision = hits/size
            # recall = self.recall(predicted, actual)
            self.metrics_mean['precision'] += precision
            # self.metrics_mean['recall'] += recall
            self.metrics_mean['hits'] += hits
            # self.metrics[uid] = {'precision': precision,
            #                      # 'recall': recall
            # }

        for k in self.metrics_mean.keys():
            self.metrics_mean[k] /= len(result)
        self.save()
        pass
    def eval_metrics(self, result, ground_truth):
        self.metrics.clear()
        self.metrics_mean.clear()
        lowest_value = np.min(ground_truth)
        for uid, predicted in result.items():
            predicted = predicted[:self.k]
            actual = np.nonzero(ground_truth[uid,:]>lowest_value)[0]
            hits = len(set(predicted) & set(actual))
            precision = hits/len(predicted)
            # recall = self.recall(predicted, actual)
            self.metrics_mean['precision'] += precision
            # self.metrics_mean['recall'] += recall
            self.metrics_mean['hits'] += hits
            # self.metrics[uid] = {'precision': precision,
            #                      # 'recall': recall,
            #                      ''
            # }
        for k in self.metrics_mean.keys():
            self.metrics_mean[k] /= len(result)
        self.save()

    # def hits(self, predicted, actual):
    #     return len(set(predicted) & set(actual))
    # def precision(self, predicted, actual):
    #     return len(set(predicted) & set(actual))/len(predicted)
    # def recall(self, predicted, actual):
    #     return len(set(predicted) & set(actual))/len(actual)
