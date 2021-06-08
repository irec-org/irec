import numpy as np
from util import DatasetFormatter
from value_functions import MostPopular
from tqdm import tqdm
from util import MetricsEvaluator
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
dsf = DatasetFormatter()
# dsf.get_base()
dsf = dsf.load()

model = MostPopular(dsf.matrix_users_ratings)

model.interact(dsf.test_uids)

for k in tqdm(range(1,model.interactions+1)):
    me = MetricsEvaluator(model.get_id(), k)
    me.eval_metrics(model.result, dsf.matrix_users_ratings)


metric_values = defaultdict(dict)
i= 'MostPopular'
METRIC_NAME = 'precision'
for k in tqdm(range(1,model.interactions+1)):
    me = MetricsEvaluator(model.get_id(), k)
    me = me.load()
    print(me.metrics_mean)
    metric_values[i][k] = me.metrics_mean[METRIC_NAME]

pd.DataFrame(metric_values).plot()
plt.xlabel("N")
plt.ylabel("Precision@N")
plt.show()
