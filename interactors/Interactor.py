import sys, os
sys.path.insert(0, os.path.abspath('..'))

from util import Nameable

class Interactor(Nameable):
    def __init__(self, consumption_matrix=None, interactions=20):
        self.consumption_matrix = consumption_matrix
        self.interactions = interactions
        pass

    def get_reward(self,uid,iid):
        return self.consumption_matrix[uid,iid]
        
    def interact(self):
        print(self.get_verbose_name())
        pass
