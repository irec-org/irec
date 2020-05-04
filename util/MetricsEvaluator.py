from .Saveable import Saveable
import numpy as np
from collections import defaultdict
import util.metrics as metrics
import util.util as util
import ctypes

class MetricsEvaluator(Saveable):
    METRICS_PRETTY = {'precision':'Precision','hits':'Hits','cumulative_precision':'Cumulative Precision',
                      'recall':'Recall','f1':'F1 Score','ndcg':'NDCG','ild':'ILD','epc':'EPC'}
    def __init__(self, name, k, threshold):
        super().__init__()
        self.metrics = defaultdict(dict)
        self.metrics_mean = defaultdict(float)
        self.name = name
        self.k = k
        self.threshold = threshold
        # self.consumption_matrix = consumption_matrix
        # self.items_distance = np.corrcoef(consumption_matrix.T)

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
    def eval_metrics(self, result, consumption_matrix, items_popularity, items_distance):
        self.metrics.clear()
        ground_truth = [np.nonzero(consumption_matrix[uid,:]>=self.threshold)[0]
                        for uid
                        in range(consumption_matrix.shape[0])]
        self.items_distance = items_distance
        self.items_popularity = items_popularity
        
        self_id = id(self)
        args = [(self_id,int(uid),predicted[:self.k],actual)
                for (uid, predicted),actual
                in zip(result.items(),ground_truth)]
        
        results = util.run_parallel(self.eval_user,args,use_tqdm=False)
        
        self.metrics_mean.clear()
        for metric_name in results[0].keys():
            self.metrics_mean[metric_name]=np.mean([result[metric_name] for result in results])

        del self.items_distance
        del self.items_popularity
        self.save()

    @staticmethod
    def eval_user(obj_id,uid,predicted,actual):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        metrics_values = dict()
        hits = len(set(predicted) & set(actual))
        precision = hits/len(predicted)
        recall = hits/len(actual)
        metrics_values['precision'] = precision
        metrics_values['recall'] = recall
        metrics_values['hits'] = hits
        metrics_values['ild'] = metrics.ildk(predicted,self.items_distance)
        metrics_values['epc'] = metrics.epck(actual,predicted,self.items_popularity)
        return metrics_values
