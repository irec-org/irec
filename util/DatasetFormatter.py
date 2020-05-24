import pandas as pd
import numpy as np
from collections import defaultdict
import random
import math
import time
import scipy.sparse
import os
import re
from .Saveable import Saveable

class DatasetFormatter(Saveable):
    PRETTY = {'ml_100k': 'MovieLens 100k',
              'ml_1m': 'MovieLens 1M',
              'tr_te_ml_1m': 'MovieLens 1M',
              'tr_te_good_books': 'Good Books',
              'tr_te_ciao_dvd': 'Ciao DVD',
              'tr_te_amazon': 'Amazon Kindle Store',
              'tr_te_ml_10m': 'MovieLens 10M',
              'tr_te_netflix': 'Netflix'}

    BASES_DIRS = {'ml_100k':'ml-100k/',
                  'ml_1m': 'ml-1m/',
                  'tr_te_ml_1m': 'Train-Test_ML-1M/',
                  'tr_te_good_books': 'Train-Test_Good-Books/',
                  'tr_te_ciao_dvd': 'Train-Test_Ciao-DVD/',
                  'tr_te_amazon': 'Train-Test_Amazon-Kindle-Store/',
                  'tr_te_ml_10m': 'Train-Test_ML-10M/',
                  'tr_te_netflix': 'Train-Test_Netflix/'}
                  # 'tr_te_netflix': 'Train-Test_Netflix/',}

    BASES_HANDLERS = {'ml_100k':'self.get_ml_100k()',
                      'ml_1m': 'self.get_ml_1m()',
                      'tr_te_ml_1m': 'self.get_tr_te_ml_1m()',
                      'tr_te_good_books': 'self.get_tr_te_ml_1m()',
                      'tr_te_ciao_dvd': 'self.get_tr_te_ml_1m()',
                      'tr_te_amazon': 'self.get_tr_te_ml_1m()',
                      'tr_te_ml_10m': 'self.get_tr_te_ml_1m()',
                      'tr_te_netflix': 'self.get_tr_te_netflix()'}
    SELECTION_MODEL = {
        'users_train_test': {'train_size': 0.8,'test_consumes':120},
        'users_train_test_chrono': {'train_size': 0.8,'test_consumes':1}
    }
    SELECTION_MODEL_HANDLERS = {'users_train_test': 'self.run_users_train_test()',
                                'users_train_test_chrono': 'self.run_users_train_test_chrono()'}
    
    def __init__(self,base='ml_100k',
                 selection_model='users_train_test_chrono',
                 is_spmatrix=True,
                 selection_model_parameters={}, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.base = base
        self.selection_model = selection_model
        self.is_spmatrix = is_spmatrix
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

        if self.is_spmatrix:
            self.consumption_matrix = scipy.sparse.csr_matrix((df_cons.r,(df_cons.uid,df_cons.iid)),dtype=float)
            self.consumption_time_matrix = scipy.sparse.csr_matrix((df_cons.t,(df_cons.uid,df_cons.iid)))
            self.users_start_time = df_cons.groupby('uid').min()['t'].to_numpy()
        else:
            self.consumption_matrix = np.nan_to_num(np.array(df_cons.pivot(index='uid', columns='iid', values = 'r')))
            self.consumption_time_matrix = np.array(df_cons.pivot(index='uid', columns='iid', values = 't'))
            self.consumption_time_matrix[np.isnan(self.consumption_time_matrix)] = 0
            self.consumption_time_matrix = scipy.sparse.csr_matrix(self.consumption_time_matrix,dtype=np.int32)
            self.users_start_time = np.where(self.consumption_time_matrix.A > 0,self.consumption_time_matrix.A,np.inf).min(axis=1)

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
        for i, iid in enumerate(df_cons['iid'].unique()):
            iids[iid] = i
        df_cons['iid'] = df_cons['iid'].apply(lambda x: iids[x])
        self.num_users = len(df_cons['uid'].unique())
        self.num_items = len(df_cons['iid'].unique())
        
        self.num_consumes = len(df_cons)

        df_cons['uid'] = df_cons['uid']-1

        self.users_items = df_cons

        self.consumption_matrix = np.nan_to_num(np.array(self.users_items.pivot(index='uid', columns='iid', values = 'r')))

        if self.is_spmatrix:
            self.consumption_matrix = scipy.sparse.csr_matrix((df_cons.r,(df_cons.uid,df_cons.iid)),dtype=float)
            self.consumption_time_matrix = scipy.sparse.csr_matrix((df_cons.t,(df_cons.uid,df_cons.iid)))
            self.users_start_time = df_cons.groupby('uid').min()['t'].to_numpy()
        else:
            self.consumption_matrix = np.nan_to_num(np.array(df_cons.pivot(index='uid', columns='iid', values = 'r')))
            self.consumption_time_matrix = np.array(df_cons.pivot(index='uid', columns='iid', values = 't'))
            self.consumption_time_matrix[np.isnan(self.consumption_time_matrix)] = 0
            self.consumption_time_matrix = scipy.sparse.csr_matrix(self.consumption_time_matrix,dtype=np.int32)
            self.users_start_time = np.where(self.consumption_time_matrix.A > 0,self.consumption_time_matrix.A,np.inf).min(axis=1)

    def read_ratings(self, fileName):
        file = open(fileName, "r")
        file.readline()
        numratings = np.sum([1 for line in open(fileName)])
        usersId = np.zeros(numratings, dtype=np.int32)
        itemsId = np.zeros(numratings, dtype=np.int32)
        ratings = np.zeros(numratings, dtype=np.float16)
        timestamp = np.zeros(numratings, dtype=np.int32)
        
        file = open(fileName, "r")
        file.readline()
        cont = 0
        for row in file:
            values = row.split('::')
            uid, iid,rating, ts = int(float(values[0])),int(float(values[1])),values[2], int(float(values[3].replace('\n', '')))
            usersId[cont] = uid
            itemsId[cont] = iid
            ratings[cont] = rating
            timestamp[cont] = ts
            cont += 1
        
        print(numratings,usersId[-1],itemsId[-1],ratings[-1])
        file.close()
        return usersId, itemsId, ratings, timestamp, numratings

    def get_tr_te_netflix(self):
        base_dir = self.BASES_DIRS[self.base]
        u_train, i_train, r_train, t_train, numr_train = self.read_ratings(base_dir+'train.data')
        u_test, i_test, r_test, t_test, numr_test = self.read_ratings(base_dir+'test.data')
        u_full = np.concatenate((u_train, u_test))
        i_full = np.concatenate((i_train, i_test)) 
        r_full = np.concatenate((r_train, r_test))
        t_full = np.concatenate((t_train, t_test))

        self.train_uids = np.unique(u_train)
        self.test_uids = np.unique(u_test)


        self.num_users = len(set(u_train).union(set(u_test)))
        self.num_items = len(set(i_train).union(set(i_test)))
        self.num_consumes = len(u_train) + len(u_test)
        
        print(np.max(self.train_uids),np.max(self.test_uids),self.num_users,self.num_items)
        del u_train, u_test, i_train, i_test, r_train, r_test, t_train, t_test
        
        if self.is_spmatrix:
            self.consumption_matrix = scipy.sparse.csr_matrix((r_full, (u_full, i_full)), shape=(self.num_users, self.num_items), dtype=float)
            self.consumption_time_matrix = scipy.sparse.csr_matrix((t_full, (u_full, i_full)), shape=(self.num_users, self.num_items), dtype=float)
        else:
            raise RuntimeError

    def get_tr_te_ml_1m(self):
        base_dir = self.BASES_DIRS[self.base]
        df_cons1 = pd.read_csv(base_dir+'train.data',sep='::',header=None,engine='python')
        df_cons1.columns = ['uid','iid','r','t']
        df_cons1['uid'] = df_cons1['uid'].astype(np.int32)
        df_cons1['iid'] = df_cons1['iid'].astype(np.int32)
        df_cons1['t'] = df_cons1['t'].astype(np.int32)

        df_cons2 = pd.read_csv(base_dir+'test.data',sep='::',header=None,engine='python')
        df_cons2.columns = ['uid','iid','r','t']
        df_cons2['uid'] = df_cons2['uid'].astype(np.int32)
        df_cons2['iid'] = df_cons2['iid'].astype(np.int32)
        df_cons2['t'] = df_cons2['t'].astype(np.int32)

        self.train_uids = np.unique(df_cons1['uid'])
        self.test_uids = np.unique(df_cons2['uid'])
        df_cons = df_cons1.append(df_cons2)
        del df_cons1, df_cons2


        self.num_users = len(np.unique(df_cons['uid']))
        self.num_items = len(np.unique(df_cons['iid']))
        self.num_consumes = len(df_cons)

        if self.is_spmatrix:
            self.consumption_matrix = scipy.sparse.csr_matrix((df_cons.r,(df_cons.uid,df_cons.iid)),dtype=float)
            self.consumption_time_matrix = scipy.sparse.csr_matrix((df_cons.t,(df_cons.uid,df_cons.iid)))
            self.users_start_time = df_cons.groupby('uid').min()['t'].to_numpy()
        else:
            self.consumption_matrix = np.nan_to_num(np.array(df_cons.pivot(index='uid', columns='iid', values = 'r')))
            self.consumption_time_matrix = np.array(df_cons.pivot(index='uid', columns='iid', values = 't'))
            self.consumption_time_matrix[np.isnan(self.consumption_time_matrix)] = 0
            self.consumption_time_matrix = scipy.sparse.csr_matrix(self.consumption_time_matrix,dtype=np.int32)
            self.users_start_time = np.where(self.consumption_time_matrix.A > 0,self.consumption_time_matrix.A,np.inf).min(axis=1)

    def gen_base(self):
        print("generating base")
        self.get_base()
        if not re.search('^tr_te',self.base):
            self.run_selection_model()
        self.save()
        
    def export_base(self,format='movielens'):
        full_rating_df = pd.DataFrame(np.where(self.consumption_matrix == np.min(self.consumption_matrix), np.nan,self.consumption_matrix))
        full_time_df = pd.DataFrame(np.where(self.consumption_time_matrix.A == np.min(self.consumption_time_matrix.A), np.nan,self.consumption_time_matrix.A))
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

