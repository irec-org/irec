from . import MF
import scipy.sparse
import util.metrics as metrics
import numpy as np
class SVD(MF):
    def fit(self,training_matrix):
        u, s, vt = scipy.sparse.linalg.svds(
            scipy.sparse.csr_matrix(training_matrix),
            k=self.num_lat)
        self.users_weights = u
        self.items_weights = vt.T*s

        observed_ui = (training_matrix.tocoo().row,training_matrix.tocoo().col) # itens observed by some user
        observed_ui_pair = tuple(zip(*observed_ui))

        print("Train RMSE:",metrics.rmse(training_matrix.data,self.get_sparse_matrix(self.users_weights,self.items_weights,observed_ui_pair)))
    def predict(self,X):
        return self.get_sparse_matrix(self.users_weights,self.items_weights,X)
        
