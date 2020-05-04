from collections import defaultdict

import numpy as np
import inquirer
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import interactors
from mf import ICFPMF
from util import DatasetFormatter, MetricsEvaluator, metrics

q = [
    inquirer.Checkbox('interactors',
                      message='Interactors to run',
                      choices=list(interactors.INTERACTORS.keys())
                      )
]
answers=inquirer.prompt(q)

INTERACTION_SIZE = interactors.Interactor().interaction_size
ITERATIONS = interactors.Interactor().get_iterations()

dsf = DatasetFormatter()
dsf = dsf.load()
# dsf.get_base()
KS = list(map(int,np.arange(INTERACTION_SIZE,ITERATIONS+1,step=INTERACTION_SIZE)))
mf = ICFPMF()
mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])

metrics_names = ['precision','recall','hits','ild','epc']
metric_values = defaultdict(lambda:defaultdict(dict))
for i in answers['interactors']:
    itr_class = interactors.INTERACTORS[i]
    if issubclass(itr_class, interactors.ICF):
        itr = itr_class(var=mf.var,
                                    user_lambda=mf.user_lambda,consumption_matrix=dsf.matrix_users_ratings)
                                        
    else:
        itr = itr_class(consumption_matrix=dsf.matrix_users_ratings)

    for j in tqdm(range(len(KS))):
        k = KS[j]
        me = MetricsEvaluator(itr.get_name(), k, 4)
        me = me.load()
        for metric_name in metrics_names:
            metric_values[metric_name][i][j] = me.metrics_mean[metric_name]
        
        
for metric_name in metrics_names:

    df = pd.DataFrame(metric_values[metric_name])
    # df['KS'] = KS
    # df=df.set_index('KS')
    print(df)
    df.plot()
    plt.xlabel("Interactions")
    plt.title(f"top-{5} recommendation")
    plt.ylabel(MetricsEvaluator.METRICS_PRETTY[metric_name])
    plt.savefig(f'img/plot_{metric_name}.png')
