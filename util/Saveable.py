import pickle
from .Nameable import Nameable
from .DirectoryDependent import DirectoryDependent
import os

class Saveable(Nameable, DirectoryDependent):
    def __init__(self, *args, **kwargs):
        DirectoryDependent.__init__(self)
        Nameable.__init__(self,*args,**kwargs)
    
    def filter_parameters(self,parameters):
        return super().filter_parameters({k: v for k, v in parameters.items() if k not in ['EXISTS']})

    def save(self):
        with open(f'{os.path.join(self.DIRS["state_save"],self.get_name())}.pickle', "wb") as f:
            pickle.dump(self, f)
    def load(self):
        with open(f'{os.path.join(self.DIRS["state_save"],self.get_name())}.pickle', "rb") as f:
            return pickle.load(f)
        
