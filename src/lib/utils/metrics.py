import numpy as np

import scipy.sparse
from collections import defaultdict 
np.seterr(all='raise')
def RelevanceEvaluator:
    def __init__(self):
        pass
    def is_relevant(self, reward):
        return True
def ThresholdRelevanceEvaluator:
    def __init__(self,threshold):
        self.threshold = self.threshold
    def is_relevant(self, reward):
        return reward>self.threshold
    
def Metric:
    def __init__(self,dataset,relevance_evaluator):
        self.dataset = dataset
        self.relevance_evaluator = relevance_evaluator
    def compute(self,uid):
        return None
    def update(self,uid,item,reward):
        pass

def Recall:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.users_true_positive = defaultdict(int)
        self.users_false_negative = defaultdict(int)
        for row in self.dataset.data:
            uid = int(row[0])
            reward = row[2]
            if self.relevance_evaluator.is_relevant(reward):
                self.users_false_negative[uid] += 1

    def compute(self,uid):
        return self.users_true_positive[uid]/(self.users_true_positive[uid]+self.users_false_negative[uid])

    def update(self,uid,item,reward):
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid] += 1

def Precision:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.users_true_positive = defaultdict(int)
        self.users_false_positive = defaultdict(int)

    def compute(self,uid):
        return self.users_true_positive[uid]/(self.users_true_positive[uid]+self.users_false_positive[uid])

    def update(self,uid,item,reward):
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid]+=1
        else:
            self.users_false_positive[uid]+=1

def Hits:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.users_true_positive = defaultdict(int)

    def compute(self,uid):
        return self.users_true_positive[uid]

    def update(self,uid,item,reward):
        if self.relevance_evaluator.is_relevant(reward):
            self.users_true_positive[uid]+=1

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
    
