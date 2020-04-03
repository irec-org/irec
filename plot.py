from collections import defaultdict

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

mf = ICFPMF()
mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])

METRIC_NAME = 'precision'
metric_values = defaultdict(dict)
for i in answers['interactors']:
    itr_class = interactors.INTERACTORS[i]
    if issubclass(itr_class, interactors.ICF):
        itr = itr_class(var=mf.var,
                        user_lambda=mf.user_lambda)
    else:
        itr = itr_class()
    for k in tqdm(range(1,itr.interactions+1)):
        me = MetricsEvaluator(itr.get_name(), k)
        me = me.load()
        print(me.metrics_mean)
        metric_values[i][k] = me.metrics_mean[METRIC_NAME]
        
pd.DataFrame(metric_values).plot()
plt.xlabel("N")
plt.ylabel("Precision@N")
plt.show()
