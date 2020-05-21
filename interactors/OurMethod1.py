from .Interactor import Interactor
import numpy as np
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
import ctypes
import scipy.spatial
class OurMethod1(Interactor):
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
        if not issubclass(self.__class__,Interactor):
            raise RuntimeError
        num_lat = len(self.items_latent_factors[0])
        I = np.eye(num_lat)

        user_candidate_items = np.array(list(range(len(self.items_latent_factors))))
        b = np.zeros(num_lat)
        A = I.copy()
        result = []
        old_mean = np.ones(b.shape)*1
        for i in range(self.interactions):
            mean = np.dot(np.linalg.inv(A),b)
            if uid == 2:
                print(mean)
                if i > 1:
                # print(mean)
                # print(np.linalg.norm(mean-old_mean))
                # print(np.corrcoef(mean,old_mean))
                    print(i,scipy.spatial.distance.cosine(mean,old_mean))
            items_uncertainty = self.alpha*np.sqrt(np.sum(self.items_latent_factors[user_candidate_items].dot(np.linalg.inv(A)) * self.items_latent_factors[user_candidate_items],axis=1))
            items_user_similarity = mean @ self.items_latent_factors[user_candidate_items].T
            items_score =  items_user_similarity + items_uncertainty
            best_items = user_candidate_items[np.argsort(items_score)[::-1]][:self.interaction_size]
            user_candidate_items = user_candidate_items[~np.isin(user_candidate_items,best_items)]
            result.extend(best_items)

            for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
                max_item_latent_factors = self.items_latent_factors[max_i]
                A += max_item_latent_factors[:,None].dot(max_item_latent_factors[None,:])
                if self.get_reward(uid,max_i) >= self.threshold:
                    b += self.get_reward(uid,max_i)*max_item_latent_factors
            old_mean = mean
        return result
