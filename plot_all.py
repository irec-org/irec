from collections import defaultdict

import numpy as np
import inquirer
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import interactors
from mf import ICFPMF
from util import DatasetFormatter, MetricsEvaluator, metrics

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 15
q = [
    inquirer.Checkbox('interactors',
                      message='Interactors to run',
                      choices=list(interactors.INTERACTORS.keys())
                      )
]
answers=inquirer.prompt(q)

INTERACTIONS = interactors.Interactor().interactions
INTERACTION_SIZE = interactors.Interactor().interaction_size
ITERATIONS = interactors.Interactor().get_iterations()
THRESHOLD = interactors.Interactor().threshold

dsf = DatasetFormatter()
dsf = dsf.load()
KS = list(map(int,np.arange(INTERACTION_SIZE,ITERATIONS+1,step=INTERACTION_SIZE)))
mf = ICFPMF()
mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])

metrics_names = ['precision','recall','hits','ild','epc','epd']
metric_values = defaultdict(lambda:defaultdict(dict))
for i in answers['interactors']:
    itr_class = interactors.INTERACTORS[i]
    if issubclass(itr_class, interactors.ICF):
        itr = itr_class(var=mf.var,
                                    user_lambda=mf.user_lambda,consumption_matrix=dsf.matrix_users_ratings,prefix_name=dsf.base)
                                        
    else:
        itr = itr_class(consumption_matrix=dsf.matrix_users_ratings,prefix_name=dsf.base)

    for j in tqdm(range(len(KS))):
        k = KS[j]
        me = MetricsEvaluator(itr.get_name(), k, THRESHOLD, INTERACTION_SIZE)
        me = me.load()
        for metric_name in metrics_names:
            metric_values[metric_name][i][j] = me.metrics_mean[metric_name]
        
fig, axs = plt.subplots(nrows=3,ncols=3,figsize=(18,13))
fig.suptitle(f"Top-{INTERACTION_SIZE} recommendation")
for ax,metric_name in zip(axs.flatten(),metrics_names):
    df = pd.DataFrame(metric_values[metric_name])
    # df['KS'] = KS
    # df=df.set_index('KS')
    ax.plot(df)
    ax.set_xlabel("Interactions")
    
    ax.set_ylabel(MetricsEvaluator.METRICS_PRETTY[metric_name],rotation='horizontal')
    ax.yaxis.set_label_coords(-0.1,1.02)

for ax, metric_name in zip(axs[2,:],metrics_names[:3]):
    df = pd.DataFrame(metric_values[metric_name]).cumsum()
    ax.plot(df)
    ax.set_xlabel("Interactions")
    
    ax.set_ylabel("Cum. "+MetricsEvaluator.METRICS_PRETTY[metric_name],rotation='horizontal')
    ax.yaxis.set_label_coords(-0.1,1.02)

s = fig.subplotpars
fig.legend(answers['interactors'],loc='lower center',
           bbox_to_anchor=[s.left, s.top+0.04, s.right-s.left, 0.05],
           ncol=6, mode="expand", borderaxespad=0,
           bbox_transform=fig.transFigure, fancybox=False, edgecolor="k")

plt.savefig(f'img/plot_all_{INTERACTIONS}_{INTERACTION_SIZE}.png',bbox_inches='tight')
plt.savefig(f'img/plot_all_{INTERACTIONS}_{INTERACTION_SIZE}.eps',bbox_inches='tight')
