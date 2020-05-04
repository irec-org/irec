import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np

from util import Nameable
from util import DirectoryDependent, Singleton
from collections import defaultdict
import pickle
import json

class Interactor(Nameable, DirectoryDependent):
    def __init__(self, consumption_matrix=None, interactions=24, interaction_size=5, threshold=4):
        self.consumption_matrix = consumption_matrix
        self.highest_value = np.max(self.consumption_matrix)
        self.lowest_value = np.min(self.consumption_matrix)
        self.values = np.unique(self.consumption_matrix)
        self.interactions = interactions
        self.interaction_size = interaction_size
        self.result = defaultdict(list)
        self.threshold = threshold

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
        elif 'pickle' == data_type:
            pass
        else:
            print("No valid data type given! Could not load result")
