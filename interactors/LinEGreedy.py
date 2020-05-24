from .Interactor import *
import numpy as np
import random
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
import ctypes
class LinEGreedy(Interactor):
    def __init__(self, epsilon=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def interact(self, items_weights):
        super().interact()
        uids = self.test_users
        self.items_weights = items_weights
        num_users = len(uids)
        # get number of latent factors 
        num_lat = len(items_weights[0])
        I = np.eye(num_lat)
        self_id = id(self)
        with threadpool_limits(limits=1, user_api='blas'):
            args = [(self_id,int(uid),) for uid in uids]
            results = util.run_parallel(self.interact_user,args)
        for i, user_result in enumerate(results):
            self.results[uids[i]] = user_result
        self.save_results()

    def init_A(self,num_lat):
        return np.eye(num_lat)

    @staticmethod
    def interact_user(obj_id,uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        if not issubclass(self.__class__,Interactor): # DANGER CODE
            raise RuntimeError

        num_lat = len(self.items_weights[0])
        I = np.eye(num_lat)

        result = []
        user_candidate_items = list(range(len(self.items_weights)))
        b = np.zeros(num_lat)
        A = self.init_A(num_lat)
        REC_ONE = False

        for i in range(self.interactions):
            for j in range(self.interaction_size):
                mean = np.dot(np.linalg.inv(A),b)
                max_i = np.NAN

                if not REC_ONE or not(self.epsilon < np.random.rand()):
                    max_i = random.choice(user_candidate_items)
                else:
                    max_i = user_candidate_items[np.argmax(mean[None,:] @ self.items_weights[user_candidate_items].T)]

                user_candidate_items.remove(max_i)
                result.append(max_i)

            for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
                max_item_mean = self.items_weights[max_i]
                A += max_item_mean[:,None].dot(max_item_mean[None,:])
                if self.get_reward(uid,max_i) >= self.threshold:
                    b += self.get_reward(uid,max_i)*max_item_mean
                    REC_ONE = True

        return result
