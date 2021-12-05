from os.path import dirname, realpath, sep, pardir
import sys, os

sys.path.append(dirname(realpath(__file__)) + sep + pardir)

import numpy as np
import scipy.sparse
from numba import jit, prange

# from .. import irec.utils
import metrics


@jit(nopython=True, parallel=True)
def _predict_sparse(users_weights, items_weights, users_items):
    n = len(users_items[0])
    results = np.zeros(n)
    for i in prange(n):
        uid = users_items[0][i]
        iid = users_items[1][i]
        results[i] = users_weights[uid] @ items_weights[iid]
    return results


class MF:
    """MF."""

    def __init__(self, num_lat=10, *args, **kwargs):
        """__init__.

        Args:
            num_lat:
            args:
            kwargs:
        """
        del args, kwargs
        self.num_lat = num_lat
        self.users_weights = None
        self.items_weights = None

    def normalize_matrix(self, matrix):
        return matrix / np.max(matrix)

    def fit(self):
        """fit."""
        pass

    def predict_sparse(self, users_items):
        return _predict_sparse(self.users_weights, self.items_weights, users_items)

    def predict(self, X):
        """predict.

        Args:
            X:
        """
        if isinstance(X, scipy.sparse.spmatrix):
            observed_ui = (X.tocoo().row, X.tocoo().col)
            X = observed_ui
        return self.predict_sparse(X)

    def score(self, X):
        """score.

        Args:
            X:
        """
        return metrics.rmse(X.data, self.predict(X))
