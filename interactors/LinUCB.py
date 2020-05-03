from .Interactor import Interactor
import numpy as np
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
import ctypes
class LinUCB(Interactor):
    def __init__(self, alpha=0.2, zeta=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha != None:
            self.alpha = alpha
        elif zeta != None:
            self.alpha = 1+np.sqrt(np.log(2/zeta)/2)

    def interact(self, uids, items_latent_factors):
        super().interact()

        self.items_latent_factors = items_latent_factors
        num_users = len(uids)
        # get number of latent factors 

        self_id = id(self)
        with threadpool_limits(limits=1, user_api='blas'):
            args = [(self_id,int(uid),) for uid in uids]
            result = util.run_parallel(self.interact_user,args)
        for i, user_result in enumerate(result):
            self.result[uids[i]] = user_result

        self.save_result()

    @staticmethod
    def interact_user(obj_id,uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        if not issubclass(self.__class__,Interactor): # DANGER CODE
            raise RuntimeError
        num_lat = len(self.items_latent_factors[0])
        I = np.eye(num_lat)

        user_candidate_items = list(range(len(self.items_latent_factors)))
        b = np.zeros(num_lat)
        A = I
        result = []

        for i in range(self.interactions):
            for j in range(self.interaction_size):
                mean = np.dot(np.linalg.inv(A),b)
                max_i = np.NAN

                max_i = user_candidate_items[np.argmax(mean[None,:] @ self.items_latent_factors[user_candidate_items].T+\
                                                       self.alpha*np.sqrt(np.sum(self.items_latent_factors[user_candidate_items].dot(np.linalg.inv(A)) * self.items_latent_factors[user_candidate_items],axis=1)))]

                user_candidate_items.remove(max_i)
                result.append(max_i)

            for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
                if self.get_reward(uid,max_i) >= self.values[-2]:
                    max_item_latent_factors = self.items_latent_factors[max_i]
                    A += max_item_latent_factors[:,None].dot(max_item_latent_factors[None,:])
                    b += self.get_reward(uid,max_i)*max_item_latent_factors
        return result