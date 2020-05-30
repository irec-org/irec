import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import scipy.sparse

from util import Saveable
from collections import defaultdict
import pickle
import json

class Interactor(Saveable):
    # def __init__(self, train_consumption_matrix=None, test_consumption_matrix=None, interactions=5*6*10*4*2, interaction_size=5, threshold=0.0001,
    #              exit_when_consumed_all=True, results_save_relevants=True,
    #              *args, **kwargs):
    def __init__(self, train_consumption_matrix=None, test_consumption_matrix=None, interactions=100, interaction_size=5, threshold=0.0001,
                 exit_when_consumed_all=False, results_save_relevants=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.consumption_matrix = consumption_matrix
        # self.highest_value = max(np.max(train_matrix),np.max(test_matrix))
        # self.lowest_value = np.min(self.consumption_matrix)
        # self.values = np.unique(self.consumption_matrix)
        self.interactions = interactions
        self.interaction_size = interaction_size
        self.threshold = threshold
        self.results = defaultdict(list)
        self.train_consumption_matrix = train_consumption_matrix
        self.test_consumption_matrix = test_consumption_matrix
        self.exit_when_consumed_all = exit_when_consumed_all
        self.results_save_relevants = results_save_relevants

    @property
    def test_consumption_matrix(self):
        return self._test_consumption_matrix

    @test_consumption_matrix.setter
    def test_consumption_matrix(self, test_consumption_matrix):
        if isinstance(test_consumption_matrix,scipy.sparse.spmatrix):
            self.is_spmatrix = isinstance(test_consumption_matrix,scipy.sparse.spmatrix)
            self.lowest_value = np.min(test_consumption_matrix)
            self.highest_value = np.max(test_consumption_matrix)
            print("max value set to:",self.highest_value)
            print("min value set to:",self.lowest_value)
            self.test_users = np.nonzero(np.sum(test_consumption_matrix>0,axis=1).A.flatten())[0]
            self.users_num_correct_items = np.sum(test_consumption_matrix>=self.threshold,axis=1)
        self._test_consumption_matrix = test_consumption_matrix

    @property
    def train_consumption_matrix(self):
        return self._train_consumption_matrix

    @train_consumption_matrix.setter
    def train_consumption_matrix(self, train_consumption_matrix):
        if isinstance(train_consumption_matrix,scipy.sparse.spmatrix):
            self.train_users = np.nonzero(np.sum(train_consumption_matrix>0,axis=1).A.flatten())[0]
        self._train_consumption_matrix = train_consumption_matrix


    def get_iterations(self):
        return self.interactions*self.interaction_size

    def get_reward(self,uid,iid):
        return self.test_consumption_matrix[uid,iid]
        
    def interact(self):
        print(self.get_verbose_name())
        self.results.clear()
        # self.test_users = self.test_consumption_matrix
        pass

    def interact_user(self,uid):
        pass

    def filter_parameters(self,parameters):
        return super().filter_parameters({k: v for k, v in parameters.items() if k not in ['highest_value','lowest_value','threshold','is_spmatrix','exit_when_consumed_all','results_save_relevants']})

    # @staticmethod
    # def json_entry_save_format(uid, items):
    #     return json.dumps({'uid': int(uid), 'predicted': list(map(int,items))})+'\n'

    # def save_results(self,data_type='pickle'):
    #     if 'pickle' in data_type:
    #         with open(f'{os.path.join(self.DIRS["result"],self.get_name())}.pickle', "wb") as f:
    #             pickle.dump(self.results, f)
    #     if 'txt' in data_type:
    #         with open(f'{os.path.join(self.DIRS["result"],self.get_name())}.txt', "w") as f:
    #             for uid, items in self.results.items():
    #                 f.write(self.json_entry_save_format(uid,items))

    # def load_results(self,data_type='pickle'):
    #     if 'pickle' == data_type:
    #         with open(f'{os.path.join(self.DIRS["result"],self.get_name())}.pickle', "rb") as f:
    #             return pickle.load(f)
    #     elif 'txt' == data_type:
    #         print("TXT not implemented yet")
    #     else:
    #         print("No valid data type given! Could not load result")
