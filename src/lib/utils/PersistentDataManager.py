import pickle
from .Parameterizable import Parameterizable
from .DirectoryDependent import DirectoryDependent
import os
import json
from . import utils
# import joblib
from os.path import sep

class PersistentDataManager(DirectoryDependent):
    def __init__(self, directory='state_save',extension_name='.pickle', *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.directory = directory
        self.extension_name = extension_name

    def get_fp(self,path):
        fp = os.path.join(self.DIRS[self.directory],path+self.extension_name)
        fp = util.repair_path_name(fp)
        return fp

    def save(self,path,data):
            
        fp = self.get_fp(path)
        util.create_path_to_file(fp)
        with open(fp, "wb") as f:
            pickle.dump(data, f)

    def load(self,path):
        fp = self.get_fp(path)
        with open(fp, "rb") as f:
            return pickle.load(f)

    def file_exists(self,path):
        fp = self.get_fp(path)
        return os.path.isfile(fp)
        
        
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
