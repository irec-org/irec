import inquirer
import interactors
import mf
from util import DatasetFormatter
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy.sparse

q = [
    inquirer.List('mf_model',
                      message='MF model to run',
                      choices=list(mf.MF_MODELS.keys())
                      )
]

answers=inquirer.prompt(q)
model_name = answers['mf_model']

dsf = DatasetFormatter()
dsf = dsf.load()

model_class=mf.MF_MODELS[model_name]
model = model_class(name_prefix=dsf.base)
if issubclass(model_class,(mf.ICFPMF,mf.ICFPMFS,mf.PMF)):
    model.load_var(dsf.train_consumption_matrix)
# model.fit(dsf.train_consumption_matrix)
model = model.load()

print(f"Using {model_name} in {dsf.base} dataset with {dsf.selection_model} selection model")

items_popularity = interactors.MostPopular.get_items_popularity(dsf.consumption_matrix,normalize=False)
top_iids_pop = list(reversed(np.argsort(items_popularity)))

items_entropy = interactors.Entropy.get_items_entropy(dsf.consumption_matrix)
top_iids_ent = list(reversed(np.argsort(items_entropy)))

items_logpopent = interactors.LogPopEnt.get_items_logpopent(items_popularity,items_entropy)
top_iids_logpopent = list(reversed(np.argsort(items_logpopent)))

num_items = 20

items_representativeness = interactors.MostRepresentative.get_items_representativeness(model.items_weights)
# items_representativeness = interactors.PopPlusEnt.get_items_popplusent(items_popularity,items_entropy)
# items_representativeness = interactors.PPELPE.get_items_ppelpe(items_popularity,items_entropy)
# items_representativeness = interactors.MostRepresentative.get_items_representativeness(model.items_weights)
# items_representativeness = interactors.HELF.get_items_helf(items_popularity,items_entropy,dsf.consumption_matrix.shape[0])
# items_representativeness = interactors.Entropy0.get_items_entropy(dsf.consumption_matrix)
# items_representativeness = interactors.MostPopular.get_items_popularity(dsf.consumption_matrix)
top_iids_rep = list(reversed(np.argsort(items_representativeness)))

rep_pop_corr = np.corrcoef(items_representativeness,items_popularity)[0,1]
rep_ent_corr = np.corrcoef(items_representativeness,items_entropy)[0,1]
rep_logpopent_corr = np.corrcoef(items_representativeness,items_logpopent)[0,1]

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
    table.append(("{}({:.2f})\t{:.2f}({})\t{}({})\t{:.2f}({})\t{:.2f}({})".format(item,
                                                        np.mean(dsf.consumption_matrix[:,item].data),
                                                      item_rep,item_rank_rep,
                                                      item_pop,item_rank_pop,
                                                      item_ent,item_rank_ent,
                                                      item_logpopent,item_rank_logpopent)).split('\t'))

print(tabulate(table, headers=['Item ID(Rat. mean)','Representativeness(Rank)','Popularity(Rank)[c:%.2f]'%(rep_pop_corr),
                               'Entropy(Rank)[c:%.2f]'%(rep_ent_corr),'LogPopEnt(Rank)[c:%.2f]'%(rep_logpopent_corr)]))
