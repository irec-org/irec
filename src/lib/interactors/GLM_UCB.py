from .ICF import ICF
import numpy as np
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
import scipy.optimize
import ctypes
class GLM_UCB(ICF):
    def __init__(self, c=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters['c'] = c

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def p(self,x):
        return x

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[2],(self.train_dataset.data[0],self.train_dataset.data[1])),(self.train_dataset.users_num,self.train_dataset.items_num))
        self.num_items = self.train_dataset.num_items

        # self.items_means = items_means

        mf_model = mf.ICFPMFS()
        mf_model.fit(self.train_consumption_matrix)
        self.items_means = mf_model.items_means
        self.num_latent_factors = len(self.items_latent_factors[0])
        self.I = np.eye(self.num_latent_factors)
        A = self.user_lambda*I
        self.As = defaultdict(lambda: np.copy(A))
        self.users_rec_rewards = defaultdict(list)
        self.users_rec_items_means = defaultdict(list)

    def error_user_weight_function(self,p,u_rec_rewards,u_rec_items_means):
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
        self.users_rec_rewards[uid]
        self.users_rec_items_means[uid]
        A = self.As[uid]
        # result = []
        # for i in range(self.interactions):
        if len(users_rec_items_means[uid]) == 0:
            p = np.zeros(num_lat)
        else:
            p = scipy.optimize.root(self.error_user_weight_function,
                                    p,
                                    (users_rec_rewards[uid],users_rec_items_means[uid])).x
        cov = np.linalg.inv(A)*self.var
        # for j in range(self.interaction_size):
        items_score = self.p(p[None,:] @ self.items_means[candidate_items].T) +\
            self.parameters['c'] * np.sqrt(np.log(self.t+1)) *\
            np.sqrt(np.sum(self.items_means[candidate_items].dot(cov) *\
                           self.items_means[candidate_items],axis=1))
        
        return items_score, None
        # user_candidate_items.remove(max_i)
        # result.append(max_i)

        # for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
    def update(self,uid,item,reward,additional_data):
        max_item_mean = self.items_means[item]
        users_rec_rewards[uid].append(reward)
        users_rec_items_means[uid].append(item)
        self.As[uid] += max_item_mean[:,None].dot(max_item_mean[None,:])
        # return result
