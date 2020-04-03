import sys, os
sys.path.insert(0, os.path.abspath('..'))

from util import Nameable
from util import DirectoryDependent
from collections import defaultdict
import pickle

class Interactor(Nameable, DirectoryDependent):
    def __init__(self, consumption_matrix=None, interactions=120):
        self.consumption_matrix = consumption_matrix
        self.interactions = interactions
        self.result = defaultdict(list)
        pass

    def get_reward(self,uid,iid):
        return self.consumption_matrix[uid,iid]
        
    def interact(self):
        print(self.get_verbose_name())
        self.result.clear()
        pass

    def save_result(self):
        with open(f'{os.path.join(self.DIRS["result"],self.get_name())}.pickle', "wb") as f:
            pickle.dump(self.result, f)

    def load_result(self):
        with open(f'{os.path.join(self.DIRS["result"],self.get_name())}.pickle', "rb") as f:
            return pickle.load(f)

