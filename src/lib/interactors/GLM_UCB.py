from .ICF import ICF
from .LinearICF import LinearICF
import numpy as np
from tqdm import tqdm
#import util
from threadpoolctl import threadpool_limits
import scipy.optimize
import ctypes
from collections import defaultdict
import joblib
import scipy
import mf
from utils.PersistentDataManager import PersistentDataManager

class GLM_UCB(LinearICF):
    def __init__(self, c=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.parameters.extend(['c'])

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def p(self,x):
        return x

    def train(self,train_dataset):
        super().train(train_dataset)
        # print(self.items_means.shape)
        # self.train_dataset = train_dataset
        # self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[:,2],(self.train_dataset.data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.num_total_users,self.train_dataset.num_total_items))
        # self.num_total_items = self.train_dataset.num_total_items

        # self.items_means = items_means

        # mf_model = mf.ICFPMFS(self.iterations,self.var,self.user_var,self.item_var,self.stop_criteria,num_lat=self.num_lat)
        # mf_model_id = joblib.hash((mf_model.get_id(),self.train_consumption_matrix))
        # pdm = PersistentDataManager('state_save')
        # if pdm.file_exists(mf_model_id):
            # mf_model = pdm.load(mf_model_id)
        # else:
            # mf_model.fit(self.train_consumption_matrix)
            # pdm.save(mf_model_id,mf_model)

        # self.items_means = mf_model.items_means
        # self.num_latent_factors = len(self.items_latent_factors[0])
        # self.I = np.eye(self.num_latent_factors)
        # A = self.get_user_lambda()*self.I
        # self.As = defaultdict(lambda: np.copy(A))
        self.users_rec_rewards = defaultdict(list)
        self.users_rec_items_means = defaultdict(list)
        self.p_vals = dict()

    def error_user_weight_function(self,p,u_rec_rewards,u_rec_items_means):
        # print(p.shape,len(u_rec_rewards),len(u_rec_items_means))
        # print(p.shape,u_rec_rewards,u_rec_items_means)
        # if len(u_rec_items_means) == 0:
            # return 0
        return np.sum(np.array(
            [(u_rec_rewards[t] - self.p(p.T @ u_rec_items_means[t]))*u_rec_items_means[t]
             for t in range(0,len(u_rec_items_means))]),0)

    def predict(self,uid,candidate_items,num_req_items):
        # self = ctypes.cast(obj_id, ctypes.py_object).value
        # if not issubclass(self.__class__,ICF): # DANGER CODE
        #     raise RuntimeError
        # num_lat = len(self.items_means[0])
        

        # user_candidate_items = list(range(len(self.items_means)))
        # # u_rec_rewards = []
        # # u_rec_items_means = []
        # self.users_rec_rewards[uid]
        # self.users_rec_items_means[uid]
        A = self.As[uid]
        # result = []
        # for i in range(self.interactions):
        if len(self.users_rec_items_means[uid]) == 0:
            self.p_vals[uid] = np.zeros(self.num_latent_factors)
        else:
            self.p_vals[uid] = scipy.optimize.root(self.error_user_weight_function,
                                    self.p_vals[uid],
                                    (self.users_rec_rewards[uid],self.users_rec_items_means[uid])).x
        cov = np.linalg.inv(A)*self.var
        # for j in range(self.interaction_size):
        items_score = self.p(self.p_vals[uid][None,:] @ self.items_means[candidate_items].T) +\
            self.c * np.sqrt(np.log(self.t+1)) *\
            np.sqrt(np.sum(self.items_means[candidate_items].dot(cov) *\
                           self.items_means[candidate_items],axis=1))
        items_score = items_score.flatten()
        return items_score, None
        # user_candidate_items.remove(max_i)
        # result.append(max_i)

        # for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
    def update(self,uid,item,reward,additional_data):
        # print(self.items_means.shape)
        max_item_mean = self.items_means[item,:]
        # print('item',item,max_item_mean.shape)
        # print(max_item_mean[:,None].shape)
        # print(max_item_mean[None,:].shape)
        self.users_rec_rewards[uid].append(reward)
        self.users_rec_items_means[uid].append(max_item_mean)
        self.As[uid] += max_item_mean[:,None].dot(max_item_mean[None,:])
        # return result
