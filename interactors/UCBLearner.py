from .Interactor import Interactor
import numpy as np
import random
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
import ctypes
from .Entropy import Entropy
from .MostPopular import MostPopular
class UCBLearner(Interactor):
    def __init__(self, stop=14, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = stop
    def interact(self, uids, items_factors):
        super().interact()
        items_entropy = Entropy.get_items_entropy(self.consumption_matrix,uids)
        items_popularity = MostPopular.get_items_popularity(self.consumption_matrix,uids)
        
        self.items_logpopent = items_entropy * np.ma.log(items_popularity+1).filled(0)
        self.items_logpopent = self.items_logpopent/np.max(self.items_logpopent)

        self.items_factors = items_factors
        num_users = len(uids)
        # get number of latent factors 
        num_lat = len(items_factors[0])
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
        limit = pow(2,stop)
        return 100*pow(2,num_items)/limit

    @staticmethod
    def interact_user(obj_id,uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        if not issubclass(self.__class__,Interactor): # DANGER CODE
            raise RuntimeError

        num_lat = len(self.items_factors[0])
        I = np.eye(num_lat)

        result = []
        user_candidate_items = list(range(len(self.items_factors)))
        b = np.zeros(num_lat)
        A = I

        nb_items = 0
        items_bias = self.items_logpopent

        for i in range(self.interactions):
            for j in range(self.interaction_size):
                mean = np.dot(np.linalg.inv(A),b)
                max_i = np.NAN
                max_item_weight = np.NAN
                max_e_reward = np.NINF

                pred_rule = mean[None,:] @ self.items_factors[user_candidate_items].T

                current_bias = items_bias[user_candidate_items] * max(1, np.max(pred_rule))
                bias = current_bias - (current_bias * self.discount_bias(nb_items,self.stop)/100)
                bias[bias<0] = 0
                max_i = user_candidate_items[np.argmax(pred_rule + bias)]

                user_candidate_items.remove(max_i)
                result.append(max_i)

            for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
                if self.get_reward(uid,max_i) >= self.threshold:
                    max_item_weight = self.items_factors[max_i]
                    A += max_item_weight[:,None].dot(max_item_weight[None,:])
                    b += self.get_reward(uid,max_i)*max_item_weight
                    nb_items += 1


        return result
