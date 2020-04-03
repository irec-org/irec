from .Saveable import Saveable

class MetricsEvaluator(Saveable):
    def __init__(self):
        self.metrics = defaultdict(dict)
        self.k = 20


    def eval_metrics(self, name, result, ground_truth):
        self.name = name
        self.metrics.clear()
        for uid, predicted in result.items():
            predicted = predicted[:self.k]
            actual = np.nonzero(ground_truth[uid,:]>=4)
            precision = self.precision(predicted, actual)
            recall = self.recall(predicted, actual)
            self.metrics[uid] = {'precision': precision,
                                 'recall': recall}
        self.save()

    def precision(self, predicted, actual):
        return len(set(predicted) & set(actual))#/len(predicted)
    def recall(self):
        return len(set(predicted) & set(actual))/len(actual)
