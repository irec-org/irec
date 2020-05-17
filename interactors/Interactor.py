import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import scipy.sparse

from util import Saveable
from collections import defaultdict
import pickle
import json

class Interactor(Saveable):
    def __init__(self, consumption_matrix=None, interactions=100, interaction_size=5, threshold=4.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.consumption_matrix = consumption_matrix
        self.highest_value = np.max(self.consumption_matrix)
        self.lowest_value = np.min(self.consumption_matrix)
        self.values = np.unique(self.consumption_matrix)
        self.interactions = interactions
        self.interaction_size = interaction_size
        self.result = defaultdict(list)
        self.threshold = threshold

    @property
    def consumption_matrix(self):
        return self._consumption_matrix

    @consumption_matrix.setter
    def consumption_matrix(self, consumption_matrix):
        self._consumption_matrix = consumption_matrix
        if issubclass(consumption_matrix.__class__,scipy.sparse.spmatrix):
            self.is_spmatrix = True
        else:
            self.is_spmatrix = False

    def get_iterations(self):
        return self.interactions*self.interaction_size

    def get_reward(self,uid,iid):
        return self.consumption_matrix[uid,iid]
        
    def interact(self):
        print(self.get_verbose_name())
        self.result.clear()
        pass

    def interact_user(self,uid):
        pass

    def filter_parameters(self,parameters):
        return super().filter_parameters({k: v for k, v in parameters.items() if k not in ['highest_value','lowest_value','threshold','is_spmatrix']})

    @staticmethod
    def json_entry_save_format(uid, items):
        return json.dumps({'uid': int(uid), 'predicted': list(map(int,items))})+'\n'

    def save_result(self,data_type='pickle'):
        if 'pickle' in data_type:
            with open(f'{os.path.join(self.DIRS["result"],self.get_name())}.pickle', "wb") as f:
                pickle.dump(self.result, f)
        if 'txt' in data_type:
            with open(f'{os.path.join(self.DIRS["result"],self.get_name())}.txt', "w") as f:
                for uid, items in self.result.items():
                    f.write(self.json_entry_save_format(uid,items))

    def load_result(self,data_type='pickle'):
        if 'pickle' == data_type:
            with open(f'{os.path.join(self.DIRS["result"],self.get_name())}.pickle', "rb") as f:
                return pickle.load(f)
        elif 'txt' == data_type:
            print("TXT not implemented yet")
        else:
            print("No valid data type given! Could not load result")
