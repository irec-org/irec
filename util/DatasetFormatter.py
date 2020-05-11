import pandas as pd
import numpy as np
from collections import defaultdict
import random
import math
import time
import scipy.sparse
import os

from .Saveable import Saveable

class DatasetFormatter(Saveable):
    PRETTY = {'ml_100k': 'MovieLens 100k','ml_1m': 'MovieLens 1M','tr_te_ml_1m': 'MovieLens 1M - T'}
    BASES_DIRS = {'ml_100k':'ml-100k/', 'ml_1m': 'ml-1m/', 'tr_te_ml_1m': 'TrainTest - ML-1M/'}
    BASES_HANDLERS = {'ml_100k':'self.get_ml_100k()',
                      'ml_1m': 'self.get_ml_1m()',
                      'tr_te_ml_1m': 'self.get_tr_te_ml_1m()'}
    SELECTION_MODEL = {
        'users_train_test': {'train_size': 0.8,'test_consumes':120},
        'users_train_test_chrono': {'train_size': 0.8,'test_consumes':1}
    }
    SELECTION_MODEL_HANDLERS = {'users_train_test': 'self.run_users_train_test()',
                                'users_train_test_chrono': 'self.run_users_train_test_chrono()'}
    def __init__(self,base='ml_100k',
                 selection_model='users_train_test',
                 selection_model_parameters={}, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.base = base
        self.selection_model = selection_model
        self.num_users = None
        self.num_items = None
        self.num_consumes = None
    
    @property
    def selection_model(self):
        return self._selection_model

    @selection_model.setter
    def selection_model(self, selection_model):
        if selection_model not in self.SELECTION_MODEL:
            self._selection_model = next(iter(self.SELECTION_MODEL))
            print(f"Base recommender not detected, using default:{self._base_rec}")
        else:
            self._selection_model = selection_model
        # parametros para o metodo base
        self.selection_model_parameters = {}
        
    @property
    def selection_model_parameters(self):
        return self._selection_model_parameters

    @selection_model_parameters.setter
    def selection_model_parameters(self, parameters):
        final_parameters = self.SELECTION_MODEL[self.selection_model]
        parameters_result = dict()
        for parameter in final_parameters:
            if parameter not in parameters:
                # default value
                source_of_parameters_value = final_parameters
            else:
                # input value
                source_of_parameters_value = parameters
            parameters_result[parameter] = source_of_parameters_value[parameter]

        self._selection_model_parameters = parameters_result

    def get_ml_100k(self):
        base_dir = self.BASES_DIRS[self.base]
        df_cons = pd.read_csv(base_dir+'u.data',sep='\t',header=None)
        df_cons.columns = ['uid','iid','r','t']
        df_cons['iid'] = df_cons['iid']-1
        df_cons['uid'] = df_cons['uid']-1

        df_cons['r'] = df_cons['r']

        df_cons = df_cons.sort_values(by='t')

        df_genre = pd.read_csv(base_dir+'u.genre',sep='|',header=None,index_col=1)
        id_to_genre = df_genre.to_dict()[0]
        
        df_item = pd.read_csv(base_dir+'u.item',sep='|',header=None, encoding="iso-8859-1")
        
        df_item.columns=["movie id","movie title","release date","video release date",
                      "IMDb URL","unknown","Action","Adventure","Animation",
                      "Children's","Comedy","Crime","Documentary","Drama","Fantasy",
                      "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi",
                      "Thriller","War","Western"]
        df_item = df_item.sort_values(by=['movie id'])
        df_item = df_item.set_index('movie id')
        df_item = df_item.iloc[:,5:]
        self.users_items = df_cons
        self.id_to_genre = id_to_genre
        self.items_genres = df_item.to_numpy()

        df_info = pd.read_csv(base_dir+'u.info',sep=' ',header=None)
        df_info= df_info.set_index(1)[0]
        self.num_users = df_info.loc['users']
        self.num_items = df_info.loc['items']
        self.num_consumes = df_info.loc['ratings']

        self.matrix_users_ratings = np.nan_to_num(np.array(df_cons.pivot(index='uid', columns='iid', values = 'r')))
        self.matrix_users_times = np.array(df_cons.pivot(index='uid', columns='iid', values = 't'))
        self.matrix_users_times[np.isnan(self.matrix_users_times)] = 0
        self.matrix_users_times = scipy.sparse.csr_matrix(self.matrix_users_times,dtype=np.int32)
        self.users_start_time = np.where(self.matrix_users_times.A > 0,self.matrix_users_times.A,np.inf).min(axis=1)

    def get_base(self):
        print(f"Loading {self.base} {self.BASES_DIRS[self.base]}")
        stime = time.time()
        result = eval(self.BASES_HANDLERS[self.base])
        print(f'Elapsed time: {time.time()-stime}s')
        return result
    
    def run_selection_model(self):
        print(f"Running selection model {self.selection_model}")
        stime = time.time()
        eval(self.SELECTION_MODEL_HANDLERS[self.selection_model])
        print(f'Elapsed time: {time.time()-stime}s')
    
    def run_users_train_test(self):
        self.num_train_users = round(self.num_users*(self.selection_model_parameters['train_size']))
        self.num_test_users = int(self.num_users-self.num_train_users)
        users_items_consumed=self.users_items.groupby('uid').count().iloc[:,0]
        test_candidate_users=list(users_items_consumed[users_items_consumed>=self.selection_model_parameters['test_consumes']].to_dict().keys())
        self.test_uids = np.array(random.sample(test_candidate_users,k=self.num_test_users))
        self.train_uids = np.array(list(set(range(self.num_users))-set(self.test_uids)))

    def run_users_train_test_chrono(self):
        self.num_train_users = round(self.num_users*(self.selection_model_parameters['train_size']))
        self.num_test_users = int(self.num_users-self.num_train_users)
        users_items_consumed=self.users_items.groupby('uid').count().iloc[:,0]
        test_candidate_users=np.array(list(users_items_consumed[users_items_consumed>=self.selection_model_parameters['test_consumes']].to_dict().keys()))
        self.test_uids = np.array(list(test_candidate_users[list(reversed(np.argsort(self.users_start_time[test_candidate_users])))])[:self.num_test_users])
        self.train_uids = np.array(list(set(range(self.num_users))-set(self.test_uids)))
        pass

    def filter_parameters(self,parameters):
        return super().filter_parameters({k: v for k, v in parameters.items() if k not in ['num_test_users','num_train_users','num_users', 'num_items', 'num_consumes']})

    def get_ml_1m(self):
        base_dir = self.BASES_DIRS[self.base]
        df_cons = pd.read_csv(base_dir+'ratings.dat',sep='::',header=None,engine='python')
        df_cons.columns = ['uid','iid','r','t']

        iids = dict()
        for i, iid in enumerate(np.sort(df_cons['iid'].unique())):
            iids[iid] = i
        df_cons['iid'] = df_cons['iid'].apply(lambda x: iids[x])
        self.num_users = len(df_cons['uid'].unique())
        self.num_items = len(df_cons['iid'].unique())
        
        self.num_consumes = len(df_cons)

        df_cons['uid'] = df_cons['uid']-1

        self.users_items = df_cons

        self.matrix_users_ratings = np.nan_to_num(np.array(self.users_items.pivot(index='uid', columns='iid', values = 'r')))

        self.matrix_users_times = np.array(df_cons.pivot(index='uid', columns='iid', values = 't'))
        self.matrix_users_times[np.isnan(self.matrix_users_times)] = 0
        self.matrix_users_times = scipy.sparse.csr_matrix(self.matrix_users_times,dtype=np.int32)
        self.users_start_time = np.where(self.matrix_users_times.A > 0,self.matrix_users_times.A,np.inf).min(axis=1)

    def get_tr_te_ml_1m(self):
        base_dir = self.BASES_DIRS[self.base]
        df_cons1 = pd.read_csv(base_dir+'trainSet_ml-1m.data',sep='::',header=None,engine='python')
        df_cons1.columns = ['uid','iid','r','t']

        df_cons1['iid'] = df_cons1['iid']
        df_cons1['uid'] = df_cons1['uid']

        df_cons2 = pd.read_csv(base_dir+'testSet_ml-1m.data',sep='::',header=None,engine='python')
        df_cons2.columns = ['uid','iid','r','t']

        df_cons2['iid'] = df_cons2['iid']
        df_cons2['uid'] = df_cons2['uid']

        df_cons = df_cons1.append(df_cons2)
        df_cons['r'] = df_cons['r']
        self.num_users = len(np.unique(df_cons['uid']))
        self.num_items = len(np.unique(df_cons['iid']))
        self.num_consumes = len(df_cons)


        self.train_uids = np.unique(df_cons1['uid'])
        self.test_uids = np.unique(df_cons2['uid'])

        self.users_items = df_cons

        self.matrix_users_ratings = np.nan_to_num(np.array(df_cons.pivot(index='uid', columns='iid', values = 'r')))

        self.matrix_users_times = np.array(df_cons.pivot(index='uid', columns='iid', values = 't'))
        self.matrix_users_times[np.isnan(self.matrix_users_times)] = 0
        self.matrix_users_times = scipy.sparse.csr_matrix(self.matrix_users_times,dtype=np.int32)
        self.users_start_time = np.where(self.matrix_users_times.A > 0,self.matrix_users_times.A,np.inf).min(axis=1)


    def gen_base(self):
        print("generating base")
        self.get_base()
        if self.base not in ['tr_te_ml_1m']:
            self.run_selection_model()
        self.save()
    def export_base(self,format='movielens'):
        full_rating_df = pd.DataFrame(np.where(self.matrix_users_ratings == np.min(self.matrix_users_ratings), np.nan,self.matrix_users_ratings))
        full_time_df = pd.DataFrame(np.where(self.matrix_users_times.A == np.min(self.matrix_users_times.A), np.nan,self.matrix_users_times.A))
        df = pd.concat([full_rating_df.stack(),full_time_df.stack()],axis=1)
        del full_rating_df
        del full_time_df
        df = df.reset_index()
        df.columns = ['uid','iid','r','t']
        print(self.get_name())
        
    def export_users_sets(self):
        with open(f'{os.path.join(self.DIRS["export"],"train_"+self.get_name())}.txt', "w") as f:
            f.write(str(self.train_uids))
        with open(f'{os.path.join(self.DIRS["export"],"test_"+self.get_name())}.txt', "w") as f:
            f.write(str(self.test_uids))

