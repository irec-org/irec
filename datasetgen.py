import pandas as pd
import numpy as np
from collections import defaultdict
import random
import math

import pmf

class DatasetFormatter():
    BASES_DIRS = {'movie_lens':'ml-100k/'}
    BASES_HANDLERS = {'movie_lens':'self.get_movie_lens()'}
    SELECTION_MODEL = {'users_train_test': {'train_size': 0.7879,'test_consumes':120}}
    SELECTION_MODEL_HANDLERS = {'users_train_test': 'self.run_users_train_test()'}
    def __init__(self,base='movie_lens',
                 selection_model='users_train_test',
                 selection_model_parameters={}):
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

    def get_movie_lens(self):
        base_dir = self.BASES_DIRS[self.base]
        df_cons = pd.read_csv(base_dir+'u.data',sep='\t',header=None)
        df_cons.columns = ['uid','iid','r','t']
        df_cons['iid'] = df_cons['iid']-1
        df_cons['uid'] = df_cons['uid']-1
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

        self.matrix_users_ratings = np.zeros((self.num_users,self.num_items))
        for index, row in self.users_items.iterrows():
            self.matrix_users_ratings[row['uid'],row['iid']] = row['r']
        # return df_cons, df_genre,df_item
    
    def get_base(self):
        return eval(self.BASES_HANDLERS[self.base])
    
    def run_selection_model(self):
        eval(self.SELECTION_MODEL_HANDLERS[self.selection_model])
        self.get_fixed_format_for_recs()
    
    def run_users_train_test(self):
        self.num_train_users = round(self.num_users*(self.selection_model_parameters['train_size']))
        self.num_test_users = int(self.num_users-self.num_train_users)
        users_items_consumed=self.users_items.groupby('uid').count().iloc[:,0]
        test_candidate_users=list(users_items_consumed[users_items_consumed>=self.selection_model_parameters['test_consumes']].to_dict().keys())
        # print(users_items_consumed)
        self.test_uids = random.choices(test_candidate_users,k=self.num_test_users)
        self.train_uids = list(set(range(self.num_users))-set(self.test_uids))
        rows_in_test = self.users_items['uid'].isin(self.test_uids)
        self.test_users_items=self.users_items[rows_in_test]
        self.train_users_items=self.users_items[~rows_in_test]
        # self.selected_test = []
        # self.selected_train = []
        pass
    def get_fixed_format_for_recs(self):
        test_users_items = defaultdict(list)
        test_users_ratings = defaultdict(list)
        for index, row in self.test_users_items.iterrows():
            test_users_items[row['uid']].append(row['iid'])
            test_users_ratings[row['uid']].append(row['r'])

        self.test_users_items = test_users_items
        self.test_users_ratings = test_users_ratings
        
        train_users_items = defaultdict(list)
        train_users_ratings = defaultdict(list)
        for index, row in self.train_users_items.iterrows():
            train_users_items[row['uid']].append(row['iid'])
            train_users_ratings[row['uid']].append(row['r'])

        
        self.train_users_items = train_users_items
        self.train_users_ratings = train_users_ratings
    
d = DatasetFormatter()
d.get_base()
d.run_selection_model()

model = pmf.ICFPMF(40,iterations=5)
model.fit(d.matrix_users_ratings[d.train_uids,:]/5)

# itmodel = ThompsonSampling()



# itmodel.interact(d.test_users_items,
#                  model.best.items_means, model.best.items_covs,
#                  model.best.var, model.best.u_lambda
#                  )        


# result = build_bpmf_model(d.matrix_users_ratings)


# pmf = PMF(40)
# pmf.fit(d.users_items)

# pmf.fit(d.matrix_users_ratings)

# observed_ui = np.nonzero(d.matrix_users_ratings) # itens observed by some user
# print(np.sqrt(np.mean((pmf.get_predicted()[observed_ui] - d.matrix_users_ratings[observed_ui])**2)))


# observed_ui = np.nonzero(d.matrix_users_ratings) # itens observed by some user


# from sklearn.decomposition import NMF
# from sklearn.decomposition import PCA


# model = PCA(n_components=10, init='random', random_state=0)
# model = PCA(40)
# W = model.fit_transform(d.matrix_users_ratings)
# H = model.components_
# print(W,H)
# print(np.sqrt(np.mean((np.dot(W,H)[observed_ui] - d.matrix_users_ratings[observed_ui])**2)))
# print(np.sqrt(nṕ.sum(pmf.get_predicted() - d.matrix_users_ratings)**2))
