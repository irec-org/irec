from irec.RelevanceEvaluator import RelevanceEvaluator
import numpy as np
import scipy.sparse
from collections import defaultdict

from typing import Any

np.seterr(all="raise")

"""Evaluation Metrics.

This module implements numerous evaluation metrics widely used in RS.
"""


class Metric:
    """Metric.

    Metrics are used to assess the performance of a recommendation system.
    For this, there are several metrics capable of evaluating recommendations in different ways.
    """

    def __init__(
        self,
        ground_truth_dataset: Any,
        relevance_evaluator: RelevanceEvaluator,
    ):
        """__init__.

        Args:
            ground_truth_dataset (Any): ground_truth_dataset
            relevance_evaluator (RelevanceEvaluator): relevance_evaluator
            args:
            kwargs:
        """

        self.ground_truth_dataset = ground_truth_dataset
        self.relevance_evaluator = relevance_evaluator

    def compute(self, uid: int) -> Any:
        """compute.

        This method performs the metric calculation for a given user.

        Args:
            uid (int): uid

        Returns:
            Any:
        """

        return None

    def update_recommendation(self, uid: int, item: int, reward: float) -> None:
        """update_recommendation.

        Uses user-supplied item rating to update metric attributes.

        Args:
            uid (int): uid
            item (int): item
            reward (float): reward

        Returns:
            None:
        """
        raise NotImplementedError

    def update_consumption_history(self, uid: int, item: int, reward: float) -> None:
        """update_consumption_history.

        Update items consumed by a user.

        Args:
            uid (int): uid
            item (int): item
            reward (float): reward

        Returns:
            None:
        """

        raise NotImplementedError


class NumInteractions(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_num_interactions = defaultdict(int)

    def compute(self, uid):
        return self.users_num_interactions[uid]

    def update_recommendation(self, uid, item, reward):
        self.users_num_interactions[uid] += 1


class Entropy(Metric):
    def __init__(self, items_entropy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_num_items_recommended = defaultdict(int)
        self.users_entropy_cumulated = defaultdict(float)
        self.items_entropy = items_entropy

    def compute(self, uid):
        return self.users_entropy_cumulated[uid] / self.users_num_items_recommended[uid]

    def update_recommendation(self, uid, item, reward):
        self.users_num_items_recommended[uid] += 1
        self.users_entropy_cumulated[uid] += self.items_entropy[item]


class TopItemsMembership(Metric):
    def __init__(self, items_feature_values, top_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_num_items_recommended = defaultdict(int)
        self.users_membership_count_cumulated = defaultdict(float)
        self.items_feature_values = items_feature_values
        self.top_size = top_size
        self.top_items = set(np.argsort(items_feature_values)[::-1][: self.top_size])

    def compute(self, uid):
        return (
            self.users_membership_count_cumulated[uid]
            / self.users_num_items_recommended[uid]
        )

    def update_recommendation(self, uid, item, reward):
        self.users_num_items_recommended[uid] += 1
        if item in self.top_items:
            self.users_membership_count_cumulated[uid] += 1

class AP(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.users_true_positive = defaultdict(int)
        self.users_false_positive = defaultdict(int)
        self.users_cumulated_precision = defaultdict(float)
        self.users_num_recommendations = defaultdict(int)

    def compute(self, uid):
        return self.users_cumulated_precision[uid] / self.users_num_recommendations

    def update_recommendation(self, uid, item, reward):
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid] += 1
        else:
            self.users_false_positive[uid] += 1

        self.users_cumulated_precision[uid] += self.users_true_positive[uid] / (
            self.users_true_positive[uid] + self.users_false_positive[uid]
        )
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
    else:
        items_similarity = np.corrcoef(matrix.T)
    items_similarity = (items_similarity + 1) / 2
    return 1 - items_similarity


def epdk(actual, predicted, consumed_items, items_distance):
    if len(consumed_items) == 0:
        return 1
    rel = np.zeros(items_distance.shape[0], dtype=bool)
    rel[actual] = 1
    # distances_sum
    res = (
        rel[predicted][:, None]
        @ rel[consumed_items][None, :]
        * items_distance[predicted, :][:, consumed_items]
    )
    C = 1 / (len(predicted) * np.sum(rel[consumed_items]))
    return C * np.sum(res)


def rmse(ground_truth, predicted):
    return np.sqrt(np.mean((predicted - ground_truth) ** 2))
