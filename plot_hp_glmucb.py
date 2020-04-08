import inquirer
from collections import defaultdict
import interactors
from mf import ICFPMF
from util import DatasetFormatter
import numpy as np
from tqdm import tqdm
from util import MetricsEvaluator
import pandas as pd
import matplotlib.pyplot as plt

dsf = DatasetFormatter()
dsf = dsf.load()
# dsf.get_base()

mf = ICFPMF()
mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])

itr = interactors.LinearUCB.getInstance(var=mf.var,
                                            user_lambda=mf.user_lambda,
                                            consumption_matrix=dsf.matrix_users_ratings,
)
METRIC_NAME = 'precision'
metric_values = defaultdict(dict)
for i in np.linspace(0,1,6):
    itr.c = i

    itr.result = itr.load_result()
    for k in tqdm(range(1,itr.get_iterations()+1)):
        me = MetricsEvaluator(itr.get_name(), k)
        me = me.load()

        metric_values[i][k] = me.metrics_mean[METRIC_NAME]

pd.DataFrame(metric_values).plot()
plt.xlabel("N")
plt.ylabel("Precision@N")
plt.savefig('img/hp_glmucb.png')
