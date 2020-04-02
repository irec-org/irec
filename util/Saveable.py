import pickle
from .Nameable import Nameable
from .DirectoryDependent import DirectoryDependent
import os

class Saveable(Nameable, DirectoryDependent):
    def save(self):
        with open(f'{os.path.join(self.DIRS["state_save"],self.get_name())}', "wb") as f:
            pickle.dump(self, f)
    def load(self):
        with open(f'{os.path.join(self.DIRS["state_save"],self.get_name())}', "rb") as f:
            return pickle.load(f)
        
