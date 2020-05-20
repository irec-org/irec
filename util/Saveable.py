import pickle
from .Nameable import Nameable
from .DirectoryDependent import DirectoryDependent
import os

class Saveable(Nameable, DirectoryDependent):
    def __init__(self, directory='state_save', *args, **kwargs):
        self.directory = directory
        DirectoryDependent.__init__(self)
        Nameable.__init__(self,*args,**kwargs)
    
    def filter_parameters(self,parameters):
        return super().filter_parameters({k: v for k, v in parameters.items() if k not in ['EXISTS','directory']})

    def save(self):
        with open(f'{os.path.join(self.DIRS[self.directory],self.get_name())}.pickle', "wb") as f:
            pickle.dump(self, f)
    def load(self):
        with open(f'{os.path.join(self.DIRS[self.directory],self.get_name())}.pickle', "rb") as f:
            return pickle.load(f)
        
