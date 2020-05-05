from .Saveable import Saveable
import numpy as np
from collections import defaultdict
import util.metrics as metrics
import util.util as util
import ctypes

class MetricsEvaluator(Saveable):
    METRICS_PRETTY = {'precision':'Precision','hits':'Hits','cumulative_precision':'Cumulative Precision',
                      'recall':'Recall','f1':'F1 Score','ndcg':'NDCG','ild':'ILD','epc':'EPC'}
    
    def __init__(self, name, k, threshold, size=None):
        super().__init__()
        self.metrics = defaultdict(dict)
        self.metrics_mean = defaultdict(float)
        self.name = name
        self.k = k
        self.threshold = threshold
        self.size = size

    @staticmethod
    def get_ground_truth(consumption_matrix,threshold):
        return [np.nonzero(consumption_matrix[uid,:]>=threshold)[0]
                        for uid
                        in range(consumption_matrix.shape[0])]

    def eval_chunk_metrics(self, result, ground_truth, items_popularity, items_distance):
        self.metrics.clear()
        self.items_distance = items_distance
        self.items_popularity = items_popularity
        self_id = id(self)
        args = [(self_id,int(uid),predicted[self.k-self.size:self.k],actual,list(set(predicted[:self.k-self.size]) & set(actual)))
                for (uid, predicted),actual
                in zip(result.items(),ground_truth)]
        
        results = util.run_parallel(self.eval_chunk_user,args,use_tqdm=False)
        
        self.metrics_mean.clear()
        for metric_name in results[0].keys():
            self.metrics_mean[metric_name]=np.mean([result[metric_name] for result in results])

        del self.items_distance
        del self.items_popularity
        self.save()

    @staticmethod
    def eval_chunk_user(obj_id,uid,predicted,actual,consumed_items):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        metrics_values = dict()
        hits = len(set(predicted) & set(actual))
        precision = hits/self.size
        recall = hits/len(actual)
        metrics_values['precision'] = precision
        metrics_values['recall'] = recall
        metrics_values['hits'] = hits
        metrics_values['ild'] = metrics.ildk(predicted,self.items_distance)
        metrics_values['epc'] = metrics.epck(actual,predicted,self.items_popularity)
        metrics_values['ndcg'] = metrics.ndcgk(actual,predicted)
        metrics_values['epd'] = metrics.epdk(actual,predicted,consumed_items)
        return metrics_values

    def eval_metrics(self, result, ground_truth, items_popularity, items_distance):
        self.metrics.clear()
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
        metrics_values['ndcg'] = metrics.ndcgk(actual,predicted)
        return metrics_values
