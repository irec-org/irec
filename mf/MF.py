from util import Saveable
import numpy as np

class MF(Saveable):

    def __init__(self, num_lat=10, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.num_lat = num_lat

    def normalize_matrix(self,matrix):
        return matrix/np.max(matrix)

    def fit(self):
        print(self.get_name())
        pass

    def get_matrix(self, users_weights, items_weights):
        return users_weights @ items_weights.T

    def get_sparse_matrix(self, users_weights, items_weights, users_items_pairs):
        return np.array([users_weights[uid,:] @ items_weights[iid,:] for uid, iid in users_items_pairs]).flatten()

    def get_sparse_predicted(self,users_items_pairs):
        return self.get_sparse_matrix(self.users_weights,self.items_weights,users_items_pairs)

    def get_predicted(self):
        return self.get_matrix(self.users_weights,self.items_weights)

    def filter_parameters(self,parameters):
        return super().filter_parameters({k: v for k, v in parameters.items() if k not in ['rmse','objective_value']})
