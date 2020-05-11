from util import Saveable
import numpy as np
class MF(Saveable):

    def __init__(self, num_lat=10, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.num_lat = num_lat

    def normalize_matrix(self,matrix):
        return matrix/np.max(matrix)

    def fit(self):
        print(self.get_verbose_name())
