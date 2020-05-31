import inquirer
import os
import interactors
from mf import ICFPMF
from util import DatasetFormatter
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['font.size'] = 14


fig, axs = plt.subplots(1,3)
fig.set_size_inches(15, 3)
fig.subplots_adjust(top=0.85, wspace=0.3, hspace=0.5)
# i = 0
# for ax, base,base_name in zip(axs,['tr_te_yahoo_music','tr_te_good_books','tr_te_ml_10m'],['ML-10M','Good-Books','Yahoo-Music']):

#     ax.set_title(base_name)
#     dsf = DatasetFormatter(base=base)
#     dsf = dsf.load()

#     model = interactors.OurMethod1(name_prefix=dsf.base)

#     with open(os.path.join(dsf.DIRS['result'],"weights_"+model.get_name()+".pickle"),'rb') as f:
#         users_global_model_weights = pickle.load(f)

#     xs = np.linspace(1,users_global_model_weights.shape[1],11,dtype=int)
#     # xs = [1, 5, 10, 15, 20, 40, 60, 80, 100]
#     bp = ax.boxplot([users_global_model_weights[:,x-1] for x in xs],sym='')

#     for box in bp['boxes']:
#         box.set( color='k', linewidth=2)

#     for whisker in bp['whiskers']:
#         whisker.set(color='k', linewidth=2)

#     for cap in bp['caps']:
#         cap.set(color='k', linewidth=2)

#     for median in bp['medians']:
#         median.set(color='red', linewidth=4)
#     ax.set_xlabel("Interactions")
#     if i == 0:
#         ax.set_ylabel("Parameter \Phi$",fontsize=17)
#     ax.set_xticklabels(xs)
#     fig.savefig(os.path.join(dsf.DIRS['img'],"weights_"+model.get_name()+".png"),bbox_inches='tight')
#     i+= 1

i = 0
for ax, base,base_name in zip(axs,['tr_te_yahoo_music','tr_te_good_books','tr_te_ml_10m'],['ML-10M','Good-Books','Yahoo-Music']):

    ax.set_title(base_name)
    dsf = DatasetFormatter(base=base)
    dsf = dsf.load()

    model = interactors.OurMethod1(name_prefix=dsf.base)

    with open(os.path.join(dsf.DIRS['result'],"weights_"+model.get_name()+".pickle"),'rb') as f:
        users_global_model_weights = pickle.load(f)


    xs = np.linspace(0,users_global_model_weights.shape[1]-1,11,dtype=int)
    # xs = [1, 5, 10, 15, 20, 40, 60, 80, 100]
    ax.errorbar(xs,
                 y=np.mean(users_global_model_weights[:,xs],axis=0),
                 yerr=np.std(users_global_model_weights[:,xs],axis=0),capsize=6,
                    color='k',marker='o',markersize=6,markerfacecolor='red',linewidth=None)
    ax.set_xlabel("Interactions")
    # if i == 0:
    ax.set_ylabel("Parameter $\Phi$",fontsize=14)
    # ax.set_xticklabels(xs)
    i+= 1
    ax.set_yticks([0,0.25,0.5,0.75,1])
    ax.set_xticks(np.linspace(0,100,11,dtype=int))
    ax.set_xlim(-0.95,100.5)
    # ax.set_ylim(0,1.03)
fig.savefig(os.path.join(dsf.DIRS['img'],"weights_"+model.get_name()+".png"),bbox_inches='tight')
fig.savefig(os.path.join(dsf.DIRS['img'],"weights_"+model.get_name()+".eps"),bbox_inches='tight')
