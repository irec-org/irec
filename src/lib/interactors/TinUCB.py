from .ExperimentalInteractor import ExperimentalInteractor
import numpy as np
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
import ctypes
import functools

class TinUCB(ExperimentalInteractor):
    def __init__(self, alpha=0.2, zeta=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha != None:
            self.alpha = alpha
        elif zeta != None:
            self.alpha = 1+np.sqrt(np.log(2/zeta)/2)

    def interact(self, items_latent_factors):
        super().interact()
        uids = self.test_users

        self.items_latent_factors = items_latent_factors
        num_users = len(uids)

        self_id = id(self)
        with threadpool_limits(limits=1, user_api='blas'):
            args = [(self_id,int(uid),) for uid in uids]
            result = util.run_parallel(self.interact_user,args)
        for i, user_result in enumerate(result):
            self.result[uids[i]] = user_result

        self.save_result()

    def g(self,x):
        return x

    @staticmethod
    def interact_user(obj_id,uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        if not issubclass(self.__class__,ExperimentalInteractor):
            raise RuntimeError
        num_lat = len(self.items_latent_factors[0])
        I = np.eye(num_lat)

        user_candidate_items = np.array(list(range(len(self.items_latent_factors))))
        b = np.zeros(num_lat)
        A = I.copy()
        result = []

        A_sums_history = [A]

        for i in range(self.interactions):
            A_weights = [self.g(i+1)/self.g(len(A_sums_history)) for i in range(len(A_sums_history))] 
            
            A = functools.reduce(lambda a,b: a+b,map(lambda x,w: x*w, zip(A_sums_history,A_weights)))
            # A = functools.reduce(lambda a,b: a+b,[_A*(2**(i+1)/2**len(A_sums_history)) for i,_A in enumerate(A_sums_history)])

            
            mean = np.dot(np.linalg.inv(A),b)
            best_items = user_candidate_items[np.argsort(mean @ self.items_latent_factors[user_candidate_items].T+\
                                                         self.alpha*np.sqrt(np.sum(self.items_latent_factors[user_candidate_items].dot(np.linalg.inv(A)) * self.items_latent_factors[user_candidate_items],axis=1)))[::-1]][:self.interaction_size]

            user_candidate_items = user_candidate_items[~np.isin(user_candidate_items,best_items)]
            result.extend(best_items)

            for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
                max_item_latent_factors = self.items_latent_factors[max_i]
                # A += max_item_latent_factors[:,None].dot(max_item_latent_factors[None,:])
                A_sums_history.append(max_item_latent_factors[:,None].dot(max_item_latent_factors[None,:]))
                if self.get_reward(uid,max_i) >= self.threshold:
                    b += self.get_reward(uid,max_i)*max_item_latent_factors
        return result
