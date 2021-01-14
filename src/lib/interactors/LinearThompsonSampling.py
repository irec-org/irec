from .ICF import ICF
import numpy as np
from tqdm import tqdm
#import util
from threadpoolctl import threadpool_limits
import ctypes
from numba import jit
import scipy
import mf
import joblib

@jit(nopython=True)
def _central_limit_theorem(k):
    p = len(k)
    x = (np.sum(k) - p/2)/(np.sqrt(p/12))
    return x

@jit(nopython=True)
def _numba_multivariate_normal(mean,cov):
    n = len(mean)
    cov_eig = np.linalg.eigh(cov) # suppose that the matrix is symmetric
    x = np.zeros(n)
    for i in range(n):
        x[i] = _central_limit_theorem(np.random.uniform(0,1,200)) # best parameter is 20000 in terms of speed and accuracy in distribution sampling
    return ((np.diag(cov_eig[0])**(0.5)) @ cov_eig[1].T @ x)+mean

@jit(nopython=True)
def _sample_items_weights(user_candidate_items, items_means, items_covs):
    n= len(user_candidate_items)
    num_lat = items_means.shape[1]
    qs = np.zeros((n,num_lat))
    for i, item in enumerate(user_candidate_items):
        item_mean = items_means[item]
        item_cov = items_covs[item]
        qs[i] = _numba_multivariate_normal(item_mean,item_cov)
    return qs

class LinearThompsonSampling(ICF):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[:,2],(self.train_dataset.data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.num_users,self.train_dataset.num_items))
        self.num_items = self.train_dataset.num_items

        mf_model = mf.ICFPMFS(self.iterations,self.var,self.user_var,self.item_var,self.stop_criteria,num_lat=self.num_lat)
        mf_model_id = joblib.hash((mf_model.get_id(),self.train_consumption_matrix))
        pdm = PersistentDataManager('state_save')
        if pdm.file_exists(mf_model_id):
            mf_model = pdm.load(mf_model_id)
        else:
            mf_model.fit(self.train_consumption_matrix)
            pdm.save(mf_model_id,mf_model)

        self.items_means = mf_model.items_means
        self.items_covs = mf_model.items_covs
        self.num_latent_factors = len(self.items_latent_factors[0])

        # num_lat = len(self.items_means[0])
        self.I = np.eye(self.num_latent_factors)

        # user_candidate_items = np.array(list(range(len(self.items_means))))
        # get number of latent factors 
        bs = defaultdict(lambda: np.zeros(self.num_latent_factors))
        As = defaultdict(lambda: self.get_user_lambda()*I)
        # A = self.get_user_lambda()*I
        # result = []
        # num_correct_items = 0

        # get number of latent factors 

        # self_id = id(self)
        # with threadpool_limits(limits=1, user_api='blas'):
        #     args = [(self_id,int(uid),) for uid in uids]
        #     results = util.run_parallel(self.interact_user,args)
        # for i, user_result in enumerate(results):
        #     self.results[uids[i]] = user_result
        # self.save_results()


    # @staticmethod
    # def interact_user(obj_id, uid):
    #     self = ctypes.cast(obj_id, ctypes.py_object).value
    #     if not issubclass(self.__class__,ICF): # DANGER CODE
    #         raise RuntimeError

    def predict(self,uid,candidate_items,num_req_items):
        b = bs[uid]
        A = As[uid]

        mean = np.dot(np.linalg.inv(A),b)
        cov = np.linalg.inv(A)*self.var
        p = np.random.multivariate_normal(mean,cov)
        qs = _sample_items_weights(candidate_items,self.items_means, self.items_covs)

        items_score = p @ qs.T
        return items_score, {'qs':qs,'candidate_items':candidate_items}
        # best_items = user_candidate_items[np.argsort(items_score)[::-1]][:self.interaction_size]

        # result.extend(best_items)

    def update(self,uid,item,reward,additional_data):
        max_q = additional_data['qs'][np.argmax(item == additional_data['candidate_items']),:]
        A += max_q[:,None].dot(max_q[None,:])
        # if self.get_reward(uid,item) >= self.train_dataset.mean_rating:
        b += reward*max_q
                # num_correct_items += 1
                # if self.exit_when_consumed_all and num_correct_items == self.users_num_correct_items[uid]:
                #     print(f"Exiting user {uid} with {len(result)} items in total and {num_correct_items} correct ones")
                #     return np.array(result)

        # user_candidate_items = user_candidate_items[~np.isin(user_candidate_items,best_items)]
                    
