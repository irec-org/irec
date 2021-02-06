import numpy as np

import scipy.sparse
from collections import defaultdict
from utils.Parameterizable import Parameterizable
import time
from utils.util import run_parallel
import ctypes
# import utils.dataset as dataset
from utils import dataset
np.seterr(all='raise')


class RelevanceEvaluator(Parameterizable):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_relevant(self, reward):
        return True


class ThresholdRelevanceEvaluator:

    def __init__(self, threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def is_relevant(self, reward):
        return reward > self.threshold


class MetricsEvaluator(Parameterizable):

    def __init__(self,
                 metrics_classes=[],
                 relevance_evaluator=ThresholdRelevanceEvaluator(3.999),
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_classes = metrics_classes
        self.relevance_evaluator = relevance_evaluator

class CumulativeMetricsEvaluator(MetricsEvaluator):

    def __init__(self, buffer_size, ground_truth_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ground_truth_dataset = ground_truth_dataset
        self.ground_truth_consumption_matrix = scipy.sparse.csr_matrix(
            (self.ground_truth_dataset.data[:, 2],
             (self.ground_truth_dataset.data[:, 0],
              self.ground_truth_dataset.data[:, 1])),
            (self.ground_truth_dataset.num_total_users,
             self.ground_truth_dataset.num_total_items))
        self.buffer_size = buffer_size
        self.parameters.extend(['buffer_size'])

    @staticmethod
    def _metric_evaluation(obj_id, metric_class):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        if issubclass(metric_class, Recall):
            metric = metric_class(
                users_false_negative=self.users_false_negative,
                ground_truth_dataset=self.ground_truth_dataset,
                relevance_evaluator=self.relevance_evaluator)
        else:
            metric = metric_class(
                ground_truth_dataset=self.ground_truth_dataset,
                relevance_evaluator=self.relevance_evaluator)

        start = 0
        start_time = time.time()
        metric_values = []
        while start < len(self.results):
            for i in range(start,
                           min(start + self.buffer_size, len(self.results))):
                uid = self.results[i][0]
                item = self.results[i][1]
                metric.update_recommendation(
                    uid, item, self.ground_truth_consumption_matrix[uid, item])
            start = min(start + self.buffer_size, len(self.results))
            metric_values.append(
                np.mean([metric.compute(uid) for uid in self.uids]))
        print(
            f"{self.__class__.__name__} spent {time.time()-start_time:.2f} seconds executing {metric_class.__name__} metric"
        )
        return metric_values

    def evaluate(self, results):
        self.users_false_negative = defaultdict(int)
        for row in self.ground_truth_dataset.data:
            uid = int(row[0])
            reward = row[2]
            if self.relevance_evaluator.is_relevant(reward):
                self.users_false_negative[uid] += 1
        uids = []
        for uid, item in results:
            uids.append(uid)
        uids = list(set(uids))
        self.uids = uids
        self.results = results
        metrics_values = defaultdict(list)
        results = run_parallel(
            self._metric_evaluation,
            [(id(self), metric_class) for metric_class in self.metrics_classes],
            use_tqdm=False)
        for result, metric_class in zip(results, self.metrics_classes):
            metrics_values[metric_class.__name__].extend(result)

        return metrics_values


class InteractionMetricsEvaluator(MetricsEvaluator):

    def __init__(self, ground_truth_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ground_truth_dataset = ground_truth_dataset
        if isinstance(ground_truth_dataset,dataset.Dataset):
            self.ground_truth_consumption_matrix = scipy.sparse.csr_matrix(
                (self.ground_truth_dataset.data[:, 2],
                 (self.ground_truth_dataset.data[:, 0],
                  self.ground_truth_dataset.data[:, 1])),
                (self.ground_truth_dataset.num_total_users,
                 self.ground_truth_dataset.num_total_items))

    @staticmethod
    def _metric_evaluation(obj_id, num_interactions, interaction_size,
                           metric_class, interactions_to_evaluate):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        start_time = time.time()
        metric_values = []
        for i in range(num_interactions):
            if issubclass(metric_class, Recall):
                metric = metric_class(
                    users_false_negative=self.users_false_negative,
                    ground_truth_dataset=self.ground_truth_dataset,
                    relevance_evaluator=self.relevance_evaluator)
            else:
                metric = metric_class(
                    ground_truth_dataset=self.ground_truth_dataset,
                    relevance_evaluator=self.relevance_evaluator)
            for uid in self.uids:
                interaction_results = self.users_items_recommended[
                    uid][i * interaction_size:i * interaction_size +
                         interaction_size]
                for item in interaction_results:
                    metric.update_recommendation(
                        uid, item, self.ground_truth_consumption_matrix[uid,
                                                                        item])

            metric_values.append(
                np.mean([metric.compute(uid) for uid in self.uids]))

        print(
            f"{self.__class__.__name__} spent {time.time()-start_time:.2f} seconds executing {metric_class.__name__} metric"
        )
        return metric_values

    def evaluate(self, num_interactions, interaction_size, results, interactions_to_evaluate=None):
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

        metrics_values = defaultdict(list)
        results = run_parallel(
            self._metric_evaluation,
            [(id(self), num_interactions, interaction_size, metric_class, interactions_to_evaluate)
             for metric_class in self.metrics_classes],
            use_tqdm=False)
        for result, metric_class in zip(results, self.metrics_classes):
            metrics_values[metric_class.__name__].extend(result)

        return metrics_values

class CumulativeInteractionMetricsEvaluator(InteractionMetricsEvaluator):
    @staticmethod
    def metric_summarize(users_metric_values):
        return np.mean(users_metric_values)
        
    @staticmethod
    def _metric_evaluation(obj_id, num_interactions, interaction_size,
                           metric_class, interactions_to_evaluate):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        start_time = time.time()
        metric_values = []
        if issubclass(metric_class, Recall):
            metric = metric_class(
                users_false_negative=self.users_false_negative,
                ground_truth_dataset=self.ground_truth_dataset,
                relevance_evaluator=self.relevance_evaluator)
        else:
            metric = metric_class(
                ground_truth_dataset=self.ground_truth_dataset,
                relevance_evaluator=self.relevance_evaluator)
        for i in range(num_interactions):
            for uid in self.uids:
                interaction_results = self.users_items_recommended[
                    uid][i * interaction_size:i * interaction_size +
                         interaction_size]
                for item in interaction_results:
                    metric.update_recommendation(
                        uid, item, self.ground_truth_consumption_matrix[uid,
                                                                        item])

            if i in interactions_to_evaluate:
                metric_values.append(
                    self.metric_summarize([metric.compute(uid) for uid in self.uids]))

        print(
            f"{self.__class__.__name__} spent {time.time()-start_time:.2f} seconds executing {metric_class.__name__} metric"
        )
        return metric_values


class UserCumulativeInteractionMetricsEvaluator(CumulativeInteractionMetricsEvaluator):
    @staticmethod
    def metric_summarize(users_metric_values):
        return users_metric_values
    

class Metric(Parameterizable):

    def __init__(self, ground_truth_dataset, relevance_evaluator, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.ground_truth_dataset = ground_truth_dataset
        self.relevance_evaluator = relevance_evaluator
        self.parameters.extend(['relevance_evaluator'])

    def compute(self, uid):
        return None

    def update_recommendation(self, uid, item, reward):
        pass

    def update_consumption_history(self, uid, item, reward):
        pass


class Recall(Metric):

    def __init__(self, users_false_negative, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_true_positive = defaultdict(int)
        self.users_false_negative = users_false_negative

    def compute(self, uid):
        if self.users_true_positive[uid] == 0 and self.users_false_negative[
                uid] == 0:
            return 0
        return self.users_true_positive[uid] / self.users_false_negative[uid]

    def update_recommendation(self, uid, item, reward):
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid] += 1


class Precision(Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_true_positive = defaultdict(int)
        self.users_false_positive = defaultdict(int)

    def compute(self, uid):
        if self.users_true_positive[uid] == 0 and self.users_false_positive[
                uid] == 0:
            return 0
        return self.users_true_positive[uid] / (self.users_true_positive[uid] +
                                                self.users_false_positive[uid])

    def update_recommendation(self, uid, item, reward):
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid] += 1
        else:
            self.users_false_positive[uid] += 1


class Hits(Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_true_positive = defaultdict(int)

    def compute(self, uid):
        return self.users_true_positive[uid]

    def update_recommendation(self, uid, item, reward):
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid] += 1


class EPC(Metric):

    def __init__(self, items_normalized_popularity, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_num_items_recommended = defaultdict(int)
        self.users_prob_not_seen_cumulated = defaultdict(float)

    def compute(self, uid):
        C_2 = 1.0 / self.users_num_items_recommended[uid]
        sum_2 = self.users_prob_not_seen_cumulated[uid]
        EPC = C_2 * sum_2
        return EPC

    def update_recommendation(self, uid, item, reward):
        self.users_num_items_recommended[uid] += 1
        probability_seen = self.items_normalized_popularity[item]
        self.users_prob_not_seen_cumulated[uid] += 1 - probability_seen


class ILD(Metric):

    def __init__(self, items_distance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items_distance = items_distance
        self.users_items_recommended = defaultdict(list)
        self.users_local_ild = defaultdict(float)

    def compute(self, uid):
        user_num_items_recommended = len(self.users_items_recommended[uid])
        if user_num_items_recommended == 0 or user_num_items_recommended == 1:
            return 1.0
        else:
            return self.users_local_ild[uid] / (
                user_num_items_recommended *
                (user_num_items_recommended - 1) / 2)

    def update_recommendation(self, uid, item, reward):
        self.users_local_ild[uid] += np.sum(
            self.items_distance[self.users_items_recommended[uid], item])
        self.users_items_recommended[uid].append(item)


class EPD:

    def __init__(self, items_distance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items_distance = items_distance
        self.users_consumed_items = defaultdict(list)
        self.users_items_recommended = defaultdict(list)
        self.users_local_ild = defaultdict(float)

        self.users_relevant_items = scipy.sparse.csr_matrix(
            self.ground_truth_dataset)
        self.users_relevant_items[self.users_relevant_items >
                                  self.ground_truth_dataset.min_rating] = True

        rel = np.zeros(self.items_distance.shape[0], dtype=bool)
        rel[actual] = 1
        # self.ground_truth_dataset.data

        # self.users_liked_items = relevance_evaluator.is_relevant()
    def compute(self, uid):
        rel = np.array(self.users_relevant_items[uid].A).flatten()
        consumed_items = self.users_consumed_items[item]
        predicted = self.users_items_recommended[uid]
        res = rel[predicted][:, None] @ rel[consumed_items][
            None, :] * self.items_distance[predicted, :][:, consumed_items]
        C = 1 / (len(predicted) * np.sum(rel[consumed_items]))
        return C * np.sum(res)

    def update_recommendation(self, uid, item, reward):
        self.users_local_ild[uid] += np.sum(
            self.items_distance[self.users_items_recommended[uid], item])
        self.users_items_recommended[uid].append(item)

    def update_consumption_history(self, uid, item, reward):
        self.users_consumed_items[uid].append(item)


class AP(Metric):

    def __init__(self, items_distance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_true_positive = defaultdict(int)
        self.users_false_positive = defaultdict(int)
        self.users_cumulated_precision = defaultdict(float)
        self.users_num_recommendations = defaultdict(int)

    def compute(self, uid):
        return self.users_cumulated_precision[
            uid] / self.users_num_recommendations

    def update_recommendation(self, uid, item, reward):
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid] += 1
        else:
            self.users_false_positive[uid] += 1

        self.users_cumulated_precision[uid] += self.users_true_positive[uid] / (
            self.users_true_positive[uid] + self.users_false_positive[uid])
        self.users_num_recommendations[uid] += 1


def mapk(actual, predicted, k):
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def hitsk(actual, predicted):
    return len(set(predicted) & set(actual))


def precisionk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(predicted)


def recallk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(actual)


def f1k(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def ndcgk(actual, predicted):
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i, p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i + 2)
        idcg += 1.0 / np.log(i + 2)
    return dcg / idcg


def epck(actual, predicted, items_popularity):
    C_2 = 1.0 / len(predicted)
    sum_2 = 0
    for i, lid in enumerate(predicted):
        # if lid in actual:
        prob_seen_k = items_popularity[lid]
        sum_2 += 1 - prob_seen_k
    EPC = C_2 * sum_2
    return EPC


def ildk(items, items_distance):
    items = np.array(items)
    num_items = len(items)
    local_ild = 0
    if num_items == 0 or num_items == 1:
        # print("Number of items:",num_items)
        return 1.0
    else:
        for i, item_1 in enumerate(items):
            for j, item_2 in enumerate(items):
                if j < i:
                    local_ild += items_distance[item_1, item_2]

    return local_ild / (num_items * (num_items - 1) / 2)


def get_items_distance(matrix):
    if isinstance(matrix, scipy.sparse.spmatrix):
        items_similarity = np.corrcoef(matrix.A.T)
        # matrix = matrix.T
        # center_matrix = matrix.sum(axis=1)
        # center_matrix = center_matrix @ center_matrix.T
        # cov_matrix = (matrix @ matrix.T - center_matrix)/(center_matrix.shape[0]-1)
        # cov_diag = np.diag(cov_matrix)
        # items_similarity = cov_matrix/np.sqrt(np.outer(cov_diag,cov_diag))
    else:
        items_similarity = np.corrcoef(matrix.T)
    # items_similarity = (items_similarity+1)/2
    items_similarity[items_similarity < 0] = 0
    return 1 - items_similarity


def epdk(actual, predicted, consumed_items, items_distance):
    if len(consumed_items) == 0:
        return 1
    rel = np.zeros(items_distance.shape[0], dtype=bool)
    rel[actual] = 1
    # distances_sum
    res = rel[predicted][:, None] @ rel[consumed_items][
        None, :] * items_distance[predicted, :][:, consumed_items]
    C = 1 / (len(predicted) * np.sum(rel[consumed_items]))
    return C * np.sum(res)


def rmse(ground_truth, predicted):
    return np.sqrt(np.mean((predicted - ground_truth)**2))
