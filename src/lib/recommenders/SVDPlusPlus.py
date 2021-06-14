from . import Recommender
import numpy as np
import mf
import ctypes
from threadpoolctl import threadpool_limits
import util


class SVDPlusPlus(Recommender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def train(self, train_matrix):
        super().train()
        self.train_matrix = train_matrix
        mf_model = mf.SVDPlusPlus()
        mf_model.fit(train_matrix)
        self.b_u, self.b_i, self.p, self.q, self.y, self.r_mean =\
            mf_model.b_u,mf_model.b_i,\
            mf_model.p,mf_model.q,\
            mf_model.y,mf_model.r_mean

    def predict(self, test_matrix):
        super().predict()
        test_uids = np.nonzero(np.sum(test_matrix > 0, axis=1).A.flatten())[0]
        self_id = id(self)

        with threadpool_limits(limits=1, user_api='blas'):
            args = [(
                self_id,
                int(uid),
            ) for uid in test_uids]
            results = util.run_parallel(self.predict_user_items, args)

        for i, user_result in enumerate(results):
            self.results[test_uids[i]] = user_result

        self.save_results()

    @staticmethod
    def predict_user_items(obj_id, uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        user_consumption = self.train_matrix[uid, :].A.flatten()
        iids = np.nonzero(user_consumption)[0]
        test_iids = np.where(user_consumption == 0)[0]
        p_u = self.p[uid] + np.sum(self.y[iids], axis=0)
        items_scores = [
            self.r_mean + self.b_u[uid] + self.b_i[iid] + p_u @ self.q[iid]
            for iid in test_iids
        ]
        top_iids = test_iids[np.argsort(items_scores)
                             [::-1]][:self.result_list_size]
        return top_iids

    def filter_parameters(self, parameters):
        return super().filter_parameters(
            {k: v
             for k, v in parameters.items() if k not in ['r_mean']})
