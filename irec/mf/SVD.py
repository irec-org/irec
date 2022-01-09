from .MF import MF
import scipy.sparse


class SVD(MF):
    def fit(self, training_matrix):
        u, s, vt = scipy.sparse.linalg.svds(
            scipy.sparse.csr_matrix(training_matrix), k=self.num_lat)
        self.users_weights = u
        self.items_weights = vt.T * s
