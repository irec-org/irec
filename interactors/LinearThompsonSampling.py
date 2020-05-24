from .ICF import ICF
import numpy as np
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
import ctypes
class LinearThompsonSampling(ICF):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def interact(self, items_means,items_covs):
        super().interact()
        uids = self.test_users
        self.items_means = items_means
        self.items_covs = items_covs
        num_users = len(uids)
        # get number of latent factors 

        self_id = id(self)
        with threadpool_limits(limits=1, user_api='blas'):
            args = [(self_id,int(uid),) for uid in uids]
            results = util.run_parallel(self.interact_user,args)
        for i, user_result in enumerate(results):
            self.results[uids[i]] = user_result
        self.save_results()

    @staticmethod
    def interact_user(obj_id, uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        if not issubclass(self.__class__,ICF): # DANGER CODE
            raise RuntimeError

        num_lat = len(self.items_means[0])
        I = np.eye(num_lat)

        user_candidate_items = list(range(len(self.items_means)))
        # get number of latent factors 
        b = np.zeros(num_lat)
        A = self.user_lambda*I
        result = []
        for i in range(self.interactions):
            tmp_max_qs = dict()
            for j in range(self.interaction_size):
                mean = np.dot(np.linalg.inv(A),b)
                cov = np.linalg.inv(A)*self.var
                p = np.random.multivariate_normal(mean,cov)
                max_i = np.NAN
                max_q = np.NAN
                max_e_reward = np.NINF
                for item in user_candidate_items:
                    item_mean = self.items_means[item]
                    item_cov = self.items_covs[item]
                    q = np.random.multivariate_normal(item_mean,item_cov)
                    e_reward = p @ q
                    if e_reward > max_e_reward:
                        max_i = item
                        max_q = q
                        max_e_reward = e_reward
                user_candidate_items.remove(max_i)
                tmp_max_qs[max_i]=max_q
                result.append(max_i)
            
            # for item in result[i*self.interaction_size:(i+1)*self.interaction_size]:
                max_q = tmp_max_qs[max_i]
                A += max_q[:,None].dot(max_q[None,:])
                if self.get_reward(uid,max_i) >= self.threshold:
                    b += self.get_reward(uid,max_i)*max_q
                    
        return result
