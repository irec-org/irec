import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np

from util import Nameable
from util import DirectoryDependent, Singleton
from collections import defaultdict
import pickle

class Interactor(Nameable, DirectoryDependent, Singleton):
    def __init__(self, consumption_matrix=None, interactions=24, interaction_size=5):
        self.consumption_matrix = consumption_matrix
        self.highest_value = np.max(self.consumption_matrix)
        self.lowest_value = np.min(self.consumption_matrix)
        self.values = np.unique(self.consumption_matrix)
        self.interactions = interactions
        self.interaction_size = interaction_size
        self.result = defaultdict(list)

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

    def save_result(self):
        with open(f'{os.path.join(self.DIRS["result"],self.get_name())}.pickle', "wb") as f:
            pickle.dump(self.result, f)

    def load_result(self):
        with open(f'{os.path.join(self.DIRS["result"],self.get_name())}.pickle', "rb") as f:
            return pickle.load(f)

