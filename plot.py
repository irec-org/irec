from collections import defaultdict

import numpy as np
import inquirer
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import interactors
from mf import ICFPMF
from util import DatasetFormatter, MetricsEvaluator


q = [
    inquirer.Checkbox('interactors',
                      message='Interactors to run',
                      choices=list(interactors.INTERACTORS.keys())
                      )
]
answers=inquirer.prompt(q)

dsf = DatasetFormatter()
dsf = dsf.load()
# dsf.get_base()
KS = list(map(int,np.arange(1,121,step=5)))
mf = ICFPMF()
mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])

metrics_names = ['precision','hits']
metric_values = defaultdict(lambda:defaultdict(dict))
for i in answers['interactors']:
    itr_class = interactors.INTERACTORS[i]
    if issubclass(itr_class, interactors.ICF):
        itr = itr_class.getInstance(var=mf.var,
                        user_lambda=mf.user_lambda)
    else:
        itr = itr_class.getInstance()

    for j in tqdm(range(len(KS))):
        k = KS[j]
        me = MetricsEvaluator(itr.get_name(), k)
        me = me.load()
        for metric_name in metrics_names:
            metric_values[metric_name][i][j] = me.metrics_mean[metric_name]
        
        
for metric_name in metrics_names:

    df = pd.DataFrame(metric_values[metric_name])
    df['KS'] = KS
    df=df.set_index('KS')
    df.plot()
    plt.xlabel("Eval. list size")
    plt.ylabel(MetricsEvaluator.METRICS_PRETTY[metric_name])
    plt.savefig(f'img/plot_{metric_name}.png')
