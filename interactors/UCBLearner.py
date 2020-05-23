from .Interactor import Interactor
import numpy as np
import random
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
import ctypes
from .Entropy import Entropy
from .MostPopular import MostPopular
from .LogPopEnt import LogPopEnt
from .PopPlusEnt import *

class UCBLearner(Interactor):
    def __init__(self, stop=14, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = stop

    def interact(self, uids, items_latent_factors):
        super().interact()
        items_entropy = Entropy.get_items_entropy(self.consumption_matrix,uids)
        items_popularity = MostPopular.get_items_popularity(self.consumption_matrix,uids,normalize=False)
        self.items_bias= LogPopEnt.get_items_logpopent(items_popularity,items_entropy)
        # self.items_bias= PopPlusEnt.get_items_popplusent(items_popularity,items_entropy)

        # items_popularity = MostPopular.get_items_popularity(self.consumption_matrix,uids,normalize=True)
        # self.items_bias = items_popularity

        self.items_latent_factors = items_latent_factors
        num_users = len(uids)
        # get number of latent factors 
        num_lat = len(items_latent_factors[0])
        I = np.eye(num_lat)
        self_id = id(self)
        with threadpool_limits(limits=1, user_api='blas'):
            args = [(self_id,int(uid),) for uid in uids]
            result = util.run_parallel(self.interact_user,args)
        for i, user_result in enumerate(result):
            self.result[uids[i]] = user_result
        self.save_result()

    @staticmethod
    def discount_bias(num_items,stop):
        limit = pow(2,stop)/100
        return pow(2,min(stop,num_items))/limit

    @staticmethod
    def interact_user(obj_id,uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        if not issubclass(self.__class__,Interactor): # DANGER CODE
            raise RuntimeError

        num_lat = len(self.items_latent_factors[0])
        I = np.eye(num_lat)

        result = []
        user_candidate_items = np.array(list(range(len(self.items_latent_factors))))
        b = np.zeros(num_lat)
        A = I

        nb_items = 0
        items_bias = self.items_bias

        num_test_items = len(np.nonzero(self.consumption_matrix[uid,:]>=self.threshold)[0])

        for i in range(self.interactions):
            mean = np.dot(np.linalg.inv(A),b)

            pred_rule = mean @ self.items_latent_factors[user_candidate_items].T

            current_bias = items_bias[user_candidate_items] * max(1, np.max(pred_rule))
            bias = current_bias - (current_bias * self.discount_bias(nb_items,self.stop)/100)
            bias[bias<0] = 0
            # bias = current_bias
            best_items = user_candidate_items[np.argsort(pred_rule + bias)[::-1]][:self.interaction_size]

            user_candidate_items = user_candidate_items[~np.isin(user_candidate_items,best_items)]
            result.extend(best_items)

            for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
                max_item_weight = self.items_latent_factors[max_i]
                A += max_item_weight[:,None].dot(max_item_weight[None,:])
                if self.get_reward(uid,max_i) >= self.threshold:
                    b += self.get_reward(uid,max_i)*max_item_weight
                    nb_items += 1
        return result
