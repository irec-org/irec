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

def epck(actual,predicted,items_popularity):
    C_2 = 1.0/len(predicted)
    sum_2=0
    for i,lid in enumerate(predicted):
        if lid in actual:
            prob_seen_k=items_popularity[lid]
            sum_2 += 1-prob_seen_k
    EPC=C_2*sum_2
    return EPC

def ildk(items,items_distance):
    items = np.array(items)
    num_items=len(items)
    local_ild=0
    if num_items == 0 or num_items == 1:
        print("Number of items:",num_items)
        return 1.0
    else:
        for i, item_1 in enumerate(items):
            for j, item_2 in enumerate(items):
                if j < i:
                    local_ild+=items_distance[item_1,item_2]

    return local_ild/(num_items*(num_items-1)/2)

def get_items_distance(matrix):
    items_distance = np.corrcoef(matrix.T)
    items_distance = (items_distance+1)/2
    return items_distance

def epdk(actual, predicted, consumed_items, items_distance):
    if len(consumed_items) == 0:
        return 1
    rel = np.zeros(items_distance.shape[0],dtype=bool)
    rel[actual] = 1
    # distances_sum
    res = rel[predicted][:,None] @ rel[consumed_items][None,:] * items_distance[predicted,:][:,consumed_items]
    C = 1/(len(predicted)*np.sum(rel[consumed_items]))
    return np.sum(res)/C
