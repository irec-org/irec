import numpy as np
np.seterr(all='raise')

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

def epck(rec_list,actual,training_matrix):

    C_2 = 1.0/len(rec_list)
    sum_2=0
    for i,lid in enumerate(rec_list):
        if lid in actual:
            prob_seen_k=np.count_nonzero(training_matrix[:,lid])/training_matrix.shape[0]
            sum_2 += 1-prob_seen_k
    EPC=C_2*sum_2
    return EPC

