import inquirer
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

for i in answers['interactors']:
    itr_class = interactors.INTERACTORS[i]
    if isinstance(itr_class, interactors.ICF):
        itr = itr_class(var=mf.var,
                        user_lambda=mf.user_lambda)
    else:
        itr = itr_class()
    itr.result = itr.load_result()
    for k in tqdm(range(1,121)):
        me = MetricsEvaluator(itr.get_name(), k)
        me.eval_metrics(itr.result, dsf.matrix_users_ratings)

