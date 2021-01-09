import pickle
from Nameable import Nameable
from DirectoryDependent import DirectoryDependent
import os
import json

class Saveable(Nameable, DirectoryDependent):
    def __init__(self, directory='state_save', *args, **kwargs):
        self.directory = directory
        DirectoryDependent.__init__(self)
        Nameable.__init__(self,*args,**kwargs)
    
    def save(self):
        with open(f'{os.path.join(self.DIRS[self.directory],self.get_name())}.pickle', "wb") as f:
            pickle.dump(self, f)
    def load(self):
        with open(f'{os.path.join(self.DIRS[self.directory],self.get_name())}.pickle', "rb") as f:
            return pickle.load(f)
        
    @staticmethod
    def json_entry_save_format(uid, items):
        return json.dumps({'uid': int(uid), 'predicted': list(map(int,items))})+'\n'

    def save_results(self,data_type='pickle'):
        if 'pickle' in data_type:
            with open(f'{os.path.join(self.DIRS["result"],self.get_name())}.pickle', "wb") as f:
                pickle.dump(self.results, f)
        if 'txt' in data_type:
            with open(f'{os.path.join(self.DIRS["result"],self.get_name())}.txt', "w") as f:
                for uid, items in self.results.items():
                    f.write(self.json_entry_save_format(uid,items))

    def load_results(self,data_type='pickle'):
        if 'pickle' == data_type:
            with open(f'{os.path.join(self.DIRS["result"],self.get_name())}.pickle', "rb") as f:
                return pickle.load(f)
        elif 'txt' == data_type:
            print("TXT not implemented yet")
        else:
            print("No valid data type given! Could not load result")
