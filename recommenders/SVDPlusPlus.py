from . import Recommender
import numpy as np
import mf
class SVDPlusPlus(Recommender):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        pass

    def train(self,train_matrix):
        super().train()
        mf_model = mf.SVDPlusPlus()
        mf_model.fit(train_matrix)
        self.b_u, self.b_i, self.p, self.q, self.y, self.r_mean =\
            mf_model.b_u,mf_model.b_i,\
            mf_model.p,mf_model.q,\
            mf_model.y,mf_model.r_mean

    def predict(self,test_matrix):
        super().predict()
        test_uids = np.nonzero(np.sum(test_matrix>0,axis=1).A.flatten())[0]

        for uid in test_uids:
            user_consumption = test_matrix[uid,:].A.flatten()

            iids = np.nonzero(user_consumption)[0]
            test_iids = np.where(user_consumption == 0)[0]
            p_u = self.p[uid] + np.sum(self.y[iids],axis=0)
            items_scores = [self.r_mean + self.b_u[uid] + self.b_i[iid] + p_u @ self.q[iid]
             for iid in test_iids]
            top_iids = test_iids[np.argsort(items_scores)[::-1]][:self.result_list_size]
            self.results[uid].extend(top_iids)

        self.save_results()

    def filter_parameters(self,parameters):
        return super().filter_parameters({k: v for k, v in parameters.items() if k not in ['r_mean']})
