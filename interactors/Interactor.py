import sys, os
sys.path.insert(0, os.path.abspath('..'))

from util import Nameable

class Interactor(Nameable):
    def __init__(self, consumption_matrix, interaction_size=5,interactions=20):
        self.consumption_matrix = consumption_matrix
        self.interaction_size = interaction_size
        self.interactions = interactions
        pass

    def get_reward(self,uid,iid):
        return comsumption_matrix[uid,iid]
        
    def interact(self,test_data):
        pass
