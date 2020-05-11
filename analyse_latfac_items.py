import inquirer
import interactors
import mf
from util import DatasetFormatter
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy.sparse

dsf = DatasetFormatter()
dsf = dsf.load()
# model = NMF(n_components=10, init='nndsvd', random_state=0)
# P = model.fit_transform(dsf.matrix_users_ratings[dsf.train_uids])
# Q = model.components_.T
# u, s, vt = scipy.sparse.linalg.svds(
#     scipy.sparse.csr_matrix(dsf.matrix_users_ratings[dsf.train_uids]),
#     k=10)
# Q = vt.T

# model = mf.PMF()
model = mf.ICFPMF()
model.load_var(dsf.matrix_users_ratings[dsf.train_uids])
model = model.load()
Q = model.items_weights




# MF_NAME = 'NMF'
# MF_NAME = 'SVD'
MF_NAME = 'ICFPMF'
# MF_NAME = 'PMF'

print(f"Using {MF_NAME} in {dsf.base} dataset with {dsf.selection_model} selection model")

items_popularity = interactors.MostPopular.get_items_popularity(dsf.matrix_users_ratings,[],normalize=False)
top_iids_pop = list(reversed(np.argsort(items_popularity)))

items_entropy = interactors.Entropy.get_items_entropy(dsf.matrix_users_ratings,[])
top_iids_ent = list(reversed(np.argsort(items_entropy)))

items_logpopent = interactors.LogPopEnt.get_items_logpopent(items_popularity,items_entropy)
top_iids_logpopent = list(reversed(np.argsort(items_logpopent)))

num_items = 20

items_representativeness = interactors.MostRepresentative.get_items_representativeness(Q)
top_iids_rep = list(reversed(np.argsort(items_representativeness)))

# print('Item ID\tRepre.(Rank)\tPop.(Rank)\tEntropy(Rank)\tLogPopEnt(Rank)')

table = []
for i in range(num_items):
    item = top_iids_rep[i]
    item_rep = items_representativeness[item]
    item_rank_rep = top_iids_rep.index(item) + 1
    item_pop = items_popularity[item]
    item_rank_pop = top_iids_pop.index(item) + 1
    item_ent = items_entropy[item]
    item_rank_ent = top_iids_ent.index(item) + 1
    item_logpopent = items_logpopent[item]
    item_rank_logpopent = top_iids_logpopent.index(item) + 1
    # print("%d\t%.2f(%d)\t%d(%d)\t%2.f(%d)\t%2.f(%d)"%(item,
    #                                                   item_rep,item_rank_ent,
    #                                                   item_pop,item_rank_pop,
    #                                                   item_ent,item_rank_ent,
    #                                                   item_logpopent,item_rank_logpopent),sep='\t')
    table.append(("{}\t{:.2f}({})\t{}({})\t{:.2f}({})\t{:.2f}({})".format(item,
                                                      item_rep,item_rank_rep,
                                                      item_pop,item_rank_pop,
                                                      item_ent,item_rank_ent,
                                                      item_logpopent,item_rank_logpopent)).split('\t'))
    
print(tabulate(table, headers=['Item ID','Representativeness(Rank)','Popularity(Rank)','Entropy(Rank)','LogPopEnt(Rank)']))
