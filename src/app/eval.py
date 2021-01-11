import inquirer
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import interactors
import mf
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
is_spmatrix = dsf.is_spmatrix

KS = list(map(int,np.arange(INTERACTION_SIZE,ITERATIONS+1,step=INTERACTION_SIZE)))

if not is_spmatrix:
    pmf_model = mf.ICFPMF(name_prefix=dsf.base)
else:
    pmf_model = mf.ICFPMFS(name_prefix=dsf.base)
pmf_model.load_var(dsf.train_consumption_matrix)

items_distance = metrics.get_items_distance(dsf.consumption_matrix)
items_popularity = interactors.MostPopular.get_items_popularity(dsf.consumption_matrix,normalize=True)
ground_truth = MetricsEvaluator.get_ground_truth(dsf.consumption_matrix,THRESHOLD)
for i in answers['interactors']:
    itr_class = interactors.INTERACTORS[i]
    if issubclass(itr_class, interactors.ICF):
        itr = itr_class(var=pmf_model.var,
                        user_lambda=pmf_model.get_user_lambda(),
                        consumption_matrix=dsf.consumption_matrix,
                        name_prefix=dsf.base
        )
    else:
        itr = itr_class(consumption_matrix=dsf.consumption_matrix,name_prefix=dsf.base)

    itr.results = itr.load_results()
    users_consumed_items = defaultdict(list)
    for j in tqdm(range(len(KS))):
        k = KS[j]
        me = MetricsEvaluator(name_suffix='interaction_%d'%(j),name=itr.get_id(), k=k,threshold=THRESHOLD)
        
        # me.eval_chunk_metrics(itr.results, dsf.consumption_matrix,5)
        tmp_results = {uid: result[j*INTERACTION_SIZE:(j+1)*INTERACTION_SIZE] for uid, result in itr.results.items()}
        me.eval_metrics(tmp_results,
                        ground_truth, items_popularity, items_distance,
                        users_consumed_items)
        for uid, items in tmp_results.items():
            users_consumed_items[uid].extend(list(set(ground_truth[uid]) & set(items)))
            

