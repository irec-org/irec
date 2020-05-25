from . import MF
import scipy.sparse
import util.metrics as metrics
class SVD(MF):
    def fit(self,training_matrix):
        u, s, vt = scipy.sparse.linalg.svds(
            scipy.sparse.csr_matrix(training_matrix),
            k=self.num_lat)
        self.users_weights = u
        self.items_weights = vt.T
        self.s = s
        

        observed_ui = (training_matrix.tocoo().row,training_matrix.tocoo().col) # itens observed by some user
        observed_ui_pair = tuple(zip(*observed_ui))

        print("RMSE:",metrics.rmse(self.get_sparse_matrix(self.users_weights*self.s,self.items_weights,observed_ui_pair),
                                   training_matrix.data))
    def predict(self,X):
        return self.get_sparse_matrix(self.users_weights*self.s,self.items_weights,X)
        
