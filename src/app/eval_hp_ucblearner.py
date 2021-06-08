import value_functions
from mf import ICFPMF
from util import DatasetFormatter
import numpy as np
import scipy.sparse
import util.metrics as metrics
from util import MetricsEvaluator
from tqdm import tqdm
INTERACTION_SIZE = value_functions.ValueFunction().interaction_size
ITERATIONS = value_functions.ValueFunction().get_iterations()
THRESHOLD = value_functions.ValueFunction().threshold
KS = list(map(int,np.arange(INTERACTION_SIZE,ITERATIONS+1,step=INTERACTION_SIZE)))
dsf = DatasetFormatter()
dsf = dsf.load()

items_distance = metrics.get_items_distance(dsf.matrix_users_ratings)
items_popularity = value_functions.MostPopular.get_items_popularity(dsf.matrix_users_ratings,[],normalize=True)
ground_truth = MetricsEvaluator.get_ground_truth(dsf.matrix_users_ratings,THRESHOLD)

itr = value_functions.UCBLearner(consumption_matrix=dsf.matrix_users_ratings,
                             prefix_name=dsf.base)
for value in range(0,51):
    itr.stop = value
    itr.result = itr.load_result()
    for j in tqdm(range(len(KS))):
        k = KS[j]
        me = MetricsEvaluator(name=itr.get_id(), k=k,threshold=THRESHOLD, interaction_size=INTERACTION_SIZE)
        me.eval_chunk_metrics(itr.result, ground_truth, items_popularity, items_distance)
