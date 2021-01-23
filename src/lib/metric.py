import numpy as np

import scipy.sparse
from collections import defaultdict 
from utils.Parameterizable import Parameterizable
np.seterr(all='raise')

class MetricsEvaluator:
    def __init__(self,metrics_classes=[]):
        self.metrics_classes = metrics_classes

class CumulativeMetricsEvaluator(MetricsEvaluator):
    NAME_ABBREVIATION = "cme"
    def __init__(self,ground_truth_dataset,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ground_truth_dataset = ground_truth_dataset
        self.ground_truth_consumption_matrix = scipy.sparse.csr_matrix((self.ground_truth_dataset.data[:,2],(self.ground_truth_dataset.data[:,0],self.ground_truth_dataset.data[:,1])),(self.ground_truth_dataset.num_total_users,self.ground_truth_dataset.num_total_items))

    def evaluate(self, results):
        uids = []
        for uid, item in results:
            uids.append(uid)
        uids = list(set(uids))
        
        metrics_values = defaultdict(list)
        for metric_class in self.metrics_classes:
            metric = metric_class(self.ground_truth_dataset,
                                  ThresholdRelevanceEvaluator(
                                      self.ground_truth_dataset.mean_rating))
            for uid, item in results:
                metric.update_recommendation(uid,item,self.ground_truth_consumption_matrix[uid,item])
                metrics_values[metric.__class__.__name__].append(np.mean([metric.compute(uid) for uid in uids]))

        return metrics_values

class InteractionMetricsEvaluator(MetricsEvaluator):
    NAME_ABBREVIATION = "ime"
    def __init__(self,ground_truth_dataset,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ground_truth_dataset = ground_truth_dataset
        self.ground_truth_consumption_matrix = scipy.sparse.csr_matrix((self.ground_truth_dataset.data[:,2],(self.ground_truth_dataset.data[:,0],self.ground_truth_dataset.data[:,1])),(self.ground_truth_dataset.num_total_users,self.ground_truth_dataset.num_total_items))

    def evaluate(self,num_interactions,interaction_size,results):
        users_items_recommended = defaultdict(list)
        for uid, item in results:
            users_items_recommended[uid].append(item)
        uids = list(users_items_recommended.keys())
        
        metrics_values = defaultdict(list)
        
        for i in range(num_interactions):
        # for uid, item in results:
            for metric_class in self.metrics_classes:
                metric = metric_class(self.ground_truth_dataset,
                    ThresholdRelevanceEvaluator(
                        self.ground_truth_dataset.mean_rating))
                for uid in uids:
                    interaction_results = users_items_recommended[uid][i:i+interaction_size]
                    for item in interaction_results:
                        metric.update_recommendation(uid,item,self.ground_truth_consumption_matrix[uid,item])

                metrics_values[metric.__class__.__name__].append(np.mean([metric.compute(uid) for uid in uids]))

        return metrics_values

class RelevanceEvaluator(Parameterizable):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def is_relevant(self, reward):
        return True

class ThresholdRelevanceEvaluator:
    def __init__(self,threshold,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.threshold = threshold
    def is_relevant(self, reward):
        return reward>self.threshold
    
class Metric(Parameterizable):
    def __init__(self,ground_truth_dataset,relevance_evaluator,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ground_truth_dataset = ground_truth_dataset
        self.relevance_evaluator = relevance_evaluator
        self.parameters.extend(['relevance_evaluator'])
    def compute(self,uid):
        return None
    def update_recommendation(self,uid,item,reward):
        pass
    def update_consumption_history(self,uid,item,reward):
        pass

class Recall(Metric):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.users_true_positive = defaultdict(int)
        self.users_false_negative = defaultdict(int)
        for row in self.ground_truth_dataset.data:
            uid = int(row[0])
            reward = row[2]
            if self.relevance_evaluator.is_relevant(reward):
                self.users_false_negative[uid] += 1

    def compute(self,uid):
        return self.users_true_positive[uid]/(self.users_true_positive[uid]+self.users_false_negative[uid])

    def update_recommendation(self,uid,item,reward):
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid] += 1

class Precision(Metric):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.users_true_positive = defaultdict(int)
        self.users_false_positive = defaultdict(int)

    def compute(self,uid):
        if self.users_true_positive[uid] == 0 and self.users_false_positive[uid] == 0:
            return 0
        return self.users_true_positive[uid]/(self.users_true_positive[uid]+self.users_false_positive[uid])

    def update_recommendation(self,uid,item,reward):
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid]+=1
        else:
            self.users_false_positive[uid]+=1

class Hits(Metric):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.users_true_positive = defaultdict(int)

    def compute(self,uid):
        return self.users_true_positive[uid]

    def update_recommendation(self,uid,item,reward):
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid]+=1

class EPC(Metric):
    def __init__(self,items_normalized_popularity,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.users_num_items_recommended = defaultdict(int)
        self.users_prob_not_seen_cumulated = defaultdict(float)

    def compute(self,uid):
        C_2 = 1.0/self.users_num_items_recommended[uid]
        sum_2 = self.users_prob_not_seen_cumulated[uid]
        EPC = C_2*sum_2
        return EPC

    def update_recommendation(self,uid,item,reward):
        self.users_num_items_recommended[uid] += 1
        probability_seen = self.items_normalized_popularity[item]
        self.users_prob_not_seen_cumulated[uid] += 1-probability_seen

class ILD(Metric):
    def __init__(self,items_distance,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.items_distance = items_distance
        self.users_items_recommended = defaultdict(list)
        self.users_local_ild = defaultdict(float)

    def compute(self,uid):
        user_num_items_recommended = len(self.users_items_recommended[uid])
        if user_num_items_recommended == 0 or user_num_items_recommended == 1:
            return 1.0
        else:
            return self.users_local_ild[uid]/(user_num_items_recommended*(user_num_items_recommended-1)/2)

    def update_recommendation(self,uid,item,reward):
        self.users_local_ild[uid] += np.sum(self.items_distance[self.users_items_recommended[uid],item])
        self.users_items_recommended[uid].append(item)

class EPD:
    def __init__(self,items_distance,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.items_distance = items_distance
        self.users_consumed_items = defaultdict(list)
        self.users_items_recommended = defaultdict(list)
        self.users_local_ild = defaultdict(float)

        self.users_relevant_items = scipy.sparse.csr_matrix(self.ground_truth_dataset)
        self.users_relevant_items[self.users_relevant_items>self.ground_truth_dataset.min_rating] = True

        rel = np.zeros(self.items_distance.shape[0],dtype=bool)
        rel[actual] = 1
        # self.ground_truth_dataset.data
        
        # self.users_liked_items = relevance_evaluator.is_relevant()
    def compute(self,uid):
        rel = np.array(self.users_relevant_items[uid].A).flatten()
        consumed_items = self.users_consumed_items[item]
        predicted = self.users_items_recommended[uid]
        res = rel[predicted][:,None] @ rel[consumed_items][None,:] * self.items_distance[predicted,:][:,consumed_items]
        C = 1/(len(predicted)*np.sum(rel[consumed_items]))
        return C*np.sum(res)

    def update_recommendation(self,uid,item,reward):
        self.users_local_ild[uid] += np.sum(self.items_distance[self.users_items_recommended[uid],item])
        self.users_items_recommended[uid].append(item)

    def update_consumption_history(self,uid,item,reward):
        self.users_consumed_items[uid].append(item)

class AP(Metric):
    def __init__(self,items_distance,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.users_true_positive = defaultdict(int)
        self.users_false_positive = defaultdict(int)
        self.users_cumulated_precision = defaultdict(float)
        self.users_num_recommendations = defaultdict(int)
    def compute(self,uid):
        return self.users_cumulated_precision[uid]/self.users_num_recommendations

    def update_recommendation(self,uid,item,reward):
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid] += 1
        else:
            self.users_false_positive[uid] += 1

        self.users_cumulated_precision[uid] += self.users_true_positive[uid]/(self.users_true_positive[uid]+self.users_false_positive[uid])
        self.users_num_recommendations[uid] += 1

def mapk(actual, predicted, k):
    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

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
    return 2*(precision*recall)/(precision+recall)

def ndcgk(actual, predicted):
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i,p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    return dcg / idcg

def epck(actual,predicted,items_popularity):
    C_2 = 1.0/len(predicted)
    sum_2=0
    for i,lid in enumerate(predicted):
        # if lid in actual:
        prob_seen_k=items_popularity[lid]
        sum_2 += 1-prob_seen_k
    EPC=C_2*sum_2
    return EPC

def ildk(items,items_distance):
    items = np.array(items)
    num_items=len(items)
    local_ild=0
    if num_items == 0 or num_items == 1:
        # print("Number of items:",num_items)
        return 1.0
    else:
        for i, item_1 in enumerate(items):
            for j, item_2 in enumerate(items):
                if j < i:
                    local_ild+=items_distance[item_1,item_2]

    return local_ild/(num_items*(num_items-1)/2)

def get_items_distance(matrix):
    if isinstance(matrix,scipy.sparse.spmatrix):
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
    items_similarity[items_similarity<0] = 0
    return 1-items_similarity

def epdk(actual, predicted, consumed_items, items_distance):
    if len(consumed_items) == 0:
        return 1
    rel = np.zeros(items_distance.shape[0],dtype=bool)
    rel[actual] = 1
    # distances_sum
    res = rel[predicted][:,None] @ rel[consumed_items][None,:] * items_distance[predicted,:][:,consumed_items]
    C = 1/(len(predicted)*np.sum(rel[consumed_items]))
    return C*np.sum(res)

def rmse(ground_truth,predicted):
    return np.sqrt(np.mean((predicted - ground_truth)**2))
    