import interactors
from mf import ICFPMF
from util import DatasetFormatter
import numpy as np
import scipy.sparse
import util.metrics as metrics
from util import MetricsEvaluator
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 15

INTERACTION_SIZE = interactors.Interactor().interaction_size
ITERATIONS = interactors.Interactor().get_iterations()
INTERACTIONS = interactors.Interactor().interactions
THRESHOLD = interactors.Interactor().threshold
KS = list(map(int,np.arange(INTERACTION_SIZE,ITERATIONS+1,step=INTERACTION_SIZE)))
dsf = DatasetFormatter()
dsf = dsf.load()

itr = interactors.UCBLearner(consumption_matrix=dsf.matrix_users_ratings,
                             prefix_name=dsf.base)

metric_values = defaultdict(lambda:defaultdict(float))

metrics_names = ['precision','recall','hits','ild','epc','epd']

for value in range(0,51):
    itr.stop = value
    
    for j in tqdm(range(len(KS))):
        k = KS[j]
        me = MetricsEvaluator(name=itr.get_id(), k=k, threshold=THRESHOLD, interaction_size=INTERACTION_SIZE)
        me = me.load()
        for metric_name in metrics_names:
            metric_values[metric_name][value] += me.metrics_mean[metric_name]

# print(metric_values)

fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(18,13))
fig.suptitle(f"Top-{INTERACTION_SIZE} recommendation with {INTERACTIONS} interactions")
for ax,metric_name in zip(axs.flatten(),metrics_names):
    x, y = np.array(list(metric_values[metric_name].keys())), np.array(list(metric_values[metric_name].values()))
    ax.plot(x,y,color='k')
    argmax = np.argmax(y)
    colors = (['k']*len(x))
    colors[argmax] = 'r'
    for i in range(len(x)):
        ax.scatter(x[i],y[i],color=colors[i],zorder=33)

    ax.set_title("Best {} at {}".format(MetricsEvaluator.METRICS_PRETTY[metric_name],argmax))
    ax.set_xlabel("Limit/Stop point")
    
    ax.set_ylabel(MetricsEvaluator.METRICS_PRETTY[metric_name],rotation='horizontal')
    ax.yaxis.set_label_coords(-0.1,1.02)

# s = fig.subplotpars
# fig.legend(answers['interactors'],loc='lower center',
#            bbox_to_anchor=[s.left, s.top+0.04, s.right-s.left, 0.05],
#            ncol=6, mode="expand", borderaxespad=0,
#            bbox_transform=fig.transFigure, fancybox=False, edgecolor="k")

plt.savefig(f'img/hp_ucblearner_{INTERACTIONS}_{INTERACTION_SIZE}.png',bbox_inches='tight')
plt.savefig(f'img/hp_ucblearner_{INTERACTIONS}_{INTERACTION_SIZE}.eps',bbox_inches='tight')
