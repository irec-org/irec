import inquirer
from tqdm import tqdm
import numpy as np

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
KS = list(map(int,np.arange(5,121,step=5)))

mf = ICFPMF()
mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
for i in answers['interactors']:
    itr_class = interactors.INTERACTORS[i]
    if issubclass(itr_class, interactors.ICF):
        itr = itr_class(var=mf.var,
                                    user_lambda=mf.user_lambda,
                                    consumption_matrix=dsf.matrix_users_ratings

        )
    else:
        itr = itr_class(consumption_matrix=dsf.matrix_users_ratings)

    itr.result = itr.load_result()
    for j in tqdm(range(len(KS))):
        k = KS[j]
        me = MetricsEvaluator(name=itr.get_name(), k=k)
        # me.eval_chunk_metrics(itr.result, dsf.matrix_users_ratings,5)
        me.eval_metrics(itr.result, dsf.matrix_users_ratings)

