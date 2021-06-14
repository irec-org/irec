import inquirer
import value_functions
from mf import ICFPMF
from util import DatasetFormatter
import numpy as np
from tqdm import tqdm
from util import MetricsEvaluator
dsf = DatasetFormatter()
dsf = dsf.load()
# dsf.get_base()

mf = ICFPMF()
mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
mf = mf.load()

itr = value_functions.LinearEGreedy(
    var=mf.var,
    user_lambda=mf.user_lambda,
    consumption_matrix=dsf.matrix_users_ratings,
)

for epsilon in np.linspace(0, 0.1, 6):
    itr.epsilon = epsilon

    itr.result = itr.load_result()
    for k in tqdm(range(1, itr.get_iterations() + 1)):
        me = MetricsEvaluator(itr.get_id(), k)
        me.eval_metrics(itr.result, dsf.matrix_users_ratings)
