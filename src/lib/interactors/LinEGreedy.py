from .ExperimentalInteractor import *
import numpy as np
import random
from tqdm import tqdm
#import util
from threadpoolctl import threadpool_limits
import ctypes
class LinEGreedy(ExperimentalInteractor):
    def __init__(self, epsilon=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.parameters.extend(['epsilon'])

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[2],(self.train_dataset.data[0],self.train_dataset.data[1])),(self.train_dataset.users_num,self.train_dataset.items_num))
        self.num_items = self.train_dataset.num_items

        mf_model = mf.SVD()
        mf_model.fit(self.train_consumption_matrix)
        self.items_weights = mf_model.items_weights


        bs = defaultdict(lambda: np.zeros(self.num_latent_factors))
        As = defaultdict(lambda: self.init_A(self.num_latent_factors))
        # uids = self.test_users
        # self.items_weights = items_weights
        # num_users = len(uids)
        # get number of latent factors 
        # num_lat = len(items_weights[0])
        # I = np.eye(num_lat)
        # self_id = id(self)
        # with threadpool_limits(limits=1, user_api='blas'):
        #     args = [(self_id,int(uid),) for uid in uids]
        #     results = util.run_parallel(self.interact_user,args)
        # for i, user_result in enumerate(results):
        #     if not self.results_save_relevants:
        #         self.results[uids[i]] = user_result
        #     else:
        #         self.results[uids[i]] = user_result[np.isin(user_result,np.nonzero(self.test_consumption_matrix[uids[i]].A.flatten())[0])]

        # self.save_results()

    def init_A(self,num_lat):
        return np.eye(num_lat)

    def predict(self,uid,candidate_items,num_req_items):
        b = bs[uid]
        A = As[uid]

        mean = np.dot(np.linalg.inv(A),b)
        max_i = np.NAN

        rand = np.random.rand(min(num_req_items,len(candidate_items)))
        rand = np.where(self.epsilon>rand, True, False) 

        cnz = np.count_nonzero(rand)
        if cnz == min(self.interaction_size,len(candidate_items)):
            items_score = mean @ self.items_weights[candidate_items].T
        else:
            items_score = np.zeros(len(candidate_items))

        randind= random.sample(list(range(len(candidate_items))),k=np.count_nonzero(rand))
        items_score[randind] = np.inf
        # rand = np.random.rand(min(self.interaction_size,len(candidate_items)))
            
        return items_score, None
    def update(self,uid,item,reward,additional_data):
        # for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
        max_item_mean = self.items_weights[item]
        A += max_item_mean[:,None].dot(max_item_mean[None,:])
        b += reward*max_item_mean

