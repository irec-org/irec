from threadpoolctl import threadpool_limits
from . import Recommender
import numpy as np
import mf
from collections import defaultdict
import util
import ctypes


class UserKNN(Recommender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def train(self, train_matrix):
        super().train()
        self.train_matrix = train_matrix
        self.num_neighbors = int(np.sqrt(train_matrix.shape[0]))
        np.seterr('warn')
        self.sim_matrix = np.corrcoef(train_matrix.A)
        np.seterr('raise')
        self.sim_matrix[np.isnan(self.sim_matrix)] = 0
        self.sim_matrix[self.sim_matrix < 0] = 0
        self.sim_matrix[self.sim_matrix == 0] = 0.001

        self.users_similar_users = dict()
        for uid in range(train_matrix.shape[0]):
            top_uids = np.argsort(self.sim_matrix[uid])[::-1]
            self.users_similar_users[uid] = top_uids[
                top_uids != uid][:self.num_neighbors]

        self.users_ratings_mean = np.zeros(train_matrix.shape[0])

        for uid in range(train_matrix.shape[0]):
            self.users_ratings_mean[uid] = np.mean(train_matrix[uid, :].data)

    def predict(self, test_matrix):
        super().predict()
        test_uids = np.nonzero(np.sum(test_matrix > 0, axis=1).A.flatten())[0]
        self_id = id(self)

        # print(len(test_matrix.data))
        # print(test_uids)

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
        train_matrix = self.train_matrix
        user_consumption = train_matrix[uid, :].A.flatten()
        consumed_iids = np.nonzero(user_consumption)[0]
        candidate_iids = np.where(user_consumption == 0)[0]

        candidate_items_score = np.zeros(candidate_iids.shape)
        neighboring_users = self.users_similar_users[uid]
        neighbors_similarities = self.sim_matrix[uid, neighboring_users]
        for i, candidate_iid in enumerate(candidate_iids):

            # consumed_neighbors = np.isin(neighboring_items,consumed_iids)
            # if np.count_nonzero(consumed_neighbors) == 0:
            #     candidate_items_score[i] = self.items_ratings_mean[candidate_iid]
            #     continue
            # neighbors_similarities[neighbors_similarities==0] = 0.001
            numerator = np.sum(neighbors_similarities *
                               (self.train_matrix[neighboring_users,
                                                  candidate_iid].A.flatten() -
                                self.users_ratings_mean[neighboring_users]))
            denominator = np.sum(neighbors_similarities)
            # try:
            candidate_items_score[
                i] = numerator / denominator + self.users_ratings_mean[uid]
            # except:
            #     print('numerator',numerator)
            #     print('denominator',denominator)
            #     print('ratings_mean',self.items_ratings_mean[candidate_iid])

        top_iids = candidate_iids[np.argsort(candidate_items_score)[::-1]]
        return top_iids[:self.result_list_size]

    def filter_parameters(self, parameters):
        return super().filter_parameters({
            k: v
            for k, v in parameters.items() if k not in ['num_neighbors']
        })
