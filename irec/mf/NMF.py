from .MF import MF
import sklearn.decomposition


class NMF(MF):
    def fit(self, training_matrix):
        model = sklearn.decomposition.NMF(n_components=self.num_lat,
                                          init='nndsvd',
                                          random_state=0)
        self.users_weights = model.fit_transform(training_matrix)
        self.items_weights = model.components_.T
