import numpy as np
import scipy
import scipy.stats
import random
from threadpoolctl import threadpool_limits
import ctypes

from irec.utils.utils import run_parallel
from mf import MF
from numba import jit


@jit(nopython=True)
def _svdplusplus(
    data,
    indptr,
    indices,
    n_u,
    num_users,
    num_items,
    num_lat,
    learn_rate,
    delta,
    delta_bias,
    bias_learn_rate,
    iterations,
    stop_criteria,
    init_mean,
    init_std,
):
    # print(n_u)
    num_r = len(data)
    r_mean = np.mean(data)
    r_std = np.std(data)
    b_u = np.zeros(num_users)
    b_i = np.zeros(num_items)
    p = np.random.normal(init_mean, init_std, (num_users, num_lat))
    q = np.random.normal(init_mean, init_std, (num_items, num_lat))

    y = np.random.normal(init_mean, init_std, (num_items, num_lat))

    rmse_old = 0.0
    for iteration in range(iterations):
        error = 0.0
        for row in range(len(indptr) - 1):
            for i in range(indptr[row], indptr[row + 1]):
                r_ui = data[i]
                uid = row
                iid = indices[i]
                y_sum = np.zeros(num_lat)
                for j in range(indptr[row], indptr[row + 1]):
                    _iid = indices[j]
                    y_sum += y[_iid]
                y_sum = y_sum / n_u[uid]

                p_u = p[uid] + y_sum

                predicted_r = r_mean + b_u[uid] + b_i[iid] + (p_u @ q[iid])
                e_ui = r_ui - predicted_r
                # raise SystemExit
                # print(e_ui)
                # print(len(e_ui))
                # print(len(e_ui))
                # print(e_ui)
                error += e_ui ** 2

                b_u[uid] += bias_learn_rate * (e_ui - delta_bias * b_u[uid])
                b_i[iid] += bias_learn_rate * (e_ui - delta_bias * b_i[iid])

                normalized_e_ui = e_ui / n_u[uid]
                p[uid] += learn_rate * (normalized_e_ui * q[iid] - delta * p[uid])
                q[iid] += learn_rate * (normalized_e_ui * p_u - delta * q[iid])
        rmse = error / num_r
        print(iteration + 1, "RMSE:", rmse)

        if np.fabs(rmse - rmse_old) <= stop_criteria:
            break
        else:
            rmse_old = rmse
    return b_u, b_i, p, q, y


class SVDPlusPlus(MF):
    def __init__(
        self,
        iterations=50,
        learn_rate=0.05,
        delta=0.015,
        delta_bias=0.002,
        bias_learn_rate=0.005,
        stop_criteria=0.009,
        init_mean=0,
        init_std=0.1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.iterations = iterations
        self.learn_rate = learn_rate
        self.delta = delta
        self.delta_bias = delta_bias
        self.bias_learn_rate = bias_learn_rate
        self.stop_criteria = stop_criteria
        self.init_mean = init_mean
        self.init_std = init_std

    def fit(self, training_matrix):
        super().fit()
        num_users = training_matrix.shape[0]
        num_items = training_matrix.shape[1]
        n_u = np.array([np.sqrt(len(i.data)) for i in training_matrix])
        self.r_mean = np.mean(training_matrix.data)
        self.b_u, self.b_i, self.p, self.q, self.y = _svdplusplus(
            training_matrix.data,
            training_matrix.indptr,
            training_matrix.indices,
            n_u,
            num_users,
            num_items,
            self.num_lat,
            self.learn_rate,
            self.delta,
            self.delta_bias_bias,
            self.bias_learn_rate,
            self.iterations,
            self.stop_criteria,
            self.init_mean,
            self.init_std,
        )
