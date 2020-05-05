import inquirer
from tqdm import tqdm
import numpy as np

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
THRESHOLD = interactors.Interactor().threshold
dsf = DatasetFormatter()
dsf = dsf.load()
# dsf.get_base()
KS = list(map(int,np.arange(INTERACTION_SIZE,ITERATIONS+1,step=INTERACTION_SIZE)))

mf = ICFPMF()
mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])

items_distance = metrics.get_items_distance(dsf.matrix_users_ratings)
items_popularity = interactors.MostPopular.get_items_popularity(dsf.matrix_users_ratings,[])
ground_truth = MetricsEvaluator.get_ground_truth(dsf.matrix_users_ratings,THRESHOLD)
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
        me = MetricsEvaluator(name=itr.get_name(), k=k,threshold=THRESHOLD, size=INTERACTION_SIZE)
        # me.eval_chunk_metrics(itr.result, dsf.matrix_users_ratings,5)
        me.eval_chunk_metrics(itr.result, ground_truth, items_popularity, items_distance)

