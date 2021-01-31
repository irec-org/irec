from .ICF import *
from .LinEGreedy import *
import numpy as np
import random
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import ctypes
import scipy
import joblib
from utils.PersistentDataManager import PersistentDataManager

class LinearEGreedy(LinEGreedy,ICF):
    def __init__(self, *args, **kwargs):
        ICF.__init__(self, *args, **kwargs)
        LinEGreedy.__init__(self, *args, **kwargs)
    
    def init_A(self,num_lat):
        return self.get_user_lambda()*np.eye(num_lat)

    def _init_items_weights(self):
        mf_model = mf.ICFPMFS(self.iterations,self.var,self.user_var,self.item_var,self.stop_criteria,num_lat=self.num_lat)
        mf_model_id = joblib.hash((mf_model.get_id(),self.train_consumption_matrix))
        pdm = PersistentDataManager('state_save')
        if pdm.file_exists(mf_model_id):
            mf_model = pdm.load(mf_model_id)
        else:
            mf_model.fit(self.train_consumption_matrix)
            pdm.save(mf_model_id,mf_model)
        
        self.items_weights = mf_model.items_means
