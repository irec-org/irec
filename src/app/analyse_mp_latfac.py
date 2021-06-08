import inquirer
import value_functions
from mf import ICFPMF
from util import DatasetFormatter
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt

dsf = DatasetFormatter()
dsf = dsf.load()
model = NMF(n_components=10, init='nndsvd', random_state=0)
P = model.fit_transform(dsf.matrix_users_ratings[dsf.train_uids])
Q = model.components_.T

items_popularity = value_functions.MostPopular.get_items_popularity(dsf.matrix_users_ratings,dsf.test_uids)

top_iids_mp = list(reversed(np.argsort(items_popularity)))

num_items = 20


top_iids_mf = list(reversed(np.argsort(np.sum(Q,axis=1))))

plt.hist(np.sum(Q,axis=1))

plt.savefig('a.png')

print('Item MP\tItem MF\tSUM\tMEAN\tMEDIAN\tMAX\tMIN\tSTD')
for i in range(num_items):
    item_mf = top_iids_mf[i]
    item_mp = top_iids_mp[i]
    print(item_mp,item_mf,"%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"%(np.sum(Q[item_mf]),
                                                                np.mean(Q[item_mf]),np.median(Q[item_mf]),np.max(Q[item_mf]),
                                                           np.min(Q[item_mf]),np.std(Q[item_mf])),sep='\t')
    print([f'{i:.2f}' for i in list(Q[item_mf])])
    
