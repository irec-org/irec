import numpy as np
from util import DatasetFormatter
from value_functions import MostPopular
from tqdm import tqdm
from util import MetricsEvaluator
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import mf
import scipy.stats

dsf = DatasetFormatter()
mf_model = mf.NMF(name_prefix=dsf.base).load()
np.seterr(divide='warn', invalid='warn')
print(mf_model.items_weights)
print(np.argmin(np.sum(mf_model.items_weights,axis=1)))
print(mf_model.items_weights/np.sum(mf_model.items_weights,axis=1)[:,None])
print(mf_model.items_weights.max())

# print(weights)
# rv = scipy.stats.norm
# print(rv.pdf(weights,np.mean(weights),np.std(weights)))
# print(weights)
