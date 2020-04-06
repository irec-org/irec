import inquirer
import interactors
from mf import ICFPMF
from util import DatasetFormatter
import numpy as np

dsf = DatasetFormatter()
# dsf = dsf.load()
dsf.get_base()

mf = ICFPMF()
mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
mf = mf.load()

itr = interactors.LinearEGreedy.getInstance(var=mf.var,
                                            user_lambda=mf.user_lambda,
                                            consumption_matrix=dsf.matrix_users_ratings,
)
for epsilon in np.linspace(0,1,11):
    itr.epsilon = epsilon
    itr.interact(dsf.test_uids, mf.items_means)
    
for epsilon in np.linspace(0,1,11):
    itr.epsilon = epsilon

    itr.result = itr.load_result()
    for k in tqdm(range(1,itr.interactions+1)):
        me = MetricsEvaluator(itr.get_name(), k)
        me.eval_metrics(itr.result, dsf.matrix_users_ratings)
