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

METRIC_NAME = 'precision'
metric_values = defaultdict(dict)
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
        print(me.metrics_mean)
        metric_values[i][j] = me.metrics_mean[METRIC_NAME]
        
pd.DataFrame(metric_values).plot()
plt.xlabel("N")
plt.ylabel("Precision")
plt.savefig('img/plot.png')
