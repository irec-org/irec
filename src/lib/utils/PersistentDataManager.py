import pickle
from Parameterizable import Parameterizable
from DirectoryDependent import DirectoryDependent
import os
import json

class PersistentDataManager(DirectoryDependent):
    def __init__(self, directory='state_save', *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.directory = directory
    
    def save(self,path,data):
        with open(f'{os.path.join(self.DIRS[self.directory],path)}.pickle', "wb") as f:
            pickle.dump(data, f)

    def load(self,path):
        with open(f'{os.path.join(self.DIRS[self.directory],path)}.pickle', "rb") as f:
            return pickle.load(f)
        
    @staticmethod
    def json_entry_save_format(uid, items):
        return json.dumps({'uid': int(uid), 'predicted': list(map(int,items))})+'\n'

    def save_results(self,path,results,data_type='pickle'):
        if 'pickle' in data_type:
            with open(f'{os.path.join(self.DIRS["result"],path)}.pickle', "wb") as f:
                pickle.dump(self.results, f)
        if 'txt' in data_type:
            with open(f'{os.path.join(self.DIRS["result"],path)}.txt', "w") as f:
                for uid, items in self.results.items():
                    f.write(self.json_entry_save_format(uid,items))

    def load_results(self,path,data_type='pickle'):
        if 'pickle' == data_type:
            with open(f'{os.path.join(self.DIRS["result"],path)}.pickle', "rb") as f:
                return pickle.load(f)
        elif 'txt' == data_type:
            print("TXT not implemented yet")
        else:
            print("No valid data type given! Could not load result")