import interactors
from mf import ICFPMF
from util import DatasetFormatter
import numpy as np
import scipy.sparse

dsf = DatasetFormatter()
dsf = dsf.load()
u, s, vt = scipy.sparse.linalg.svds(
    scipy.sparse.csr_matrix(dsf.matrix_users_ratings[dsf.train_uids]),
    k=10)
Q = vt.T
itr = interactors.UCBLearner(consumption_matrix=dsf.matrix_users_ratings,
                             prefix_name=dsf.base)
for value in range(0,51):
    itr.stop = value
    itr.interact(dsf.test_uids, Q)
