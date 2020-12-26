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
            if not self.results_save_relevants:
                self.results[uids[i]] = user_result
            else:
                self.results[uids[i]] = user_result[np.isin(user_result,np.nonzero(self.test_consumption_matrix[uids[i]].A.flatten())[0])]

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
        user_candidate_items = np.array(list(range(len(self.items_weights))))
        b = np.zeros(num_lat)
        A = self.init_A(num_lat)
        REC_ONE = False
        num_user_candidate_items = len(user_candidate_items)
        num_correct_items = 0
        for i in range(self.interactions):
            # for j in range(min(self.interaction_size,num_user_candidate_items)):
            mean = np.dot(np.linalg.inv(A),b)
            max_i = np.NAN

            items_score = mean @ self.items_weights[user_candidate_items].T
            rand = np.random.rand(min(self.interaction_size,num_user_candidate_items))
            rand = np.where(self.epsilon>rand, True, False) 
            randind= random.sample(list(range(len(user_candidate_items))),k=np.count_nonzero(rand))
            items_score[randind] = np.inf

            best_items = user_candidate_items[np.argsort(items_score)[::-1]][:self.interaction_size]
            best_items = best_items[::-1] # random itens last, NDCG DCG will benefit from this
                    
            user_candidate_items = user_candidate_items[~np.isin(user_candidate_items,best_items)]
            result.extend(best_items)

            for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
                max_item_mean = self.items_weights[max_i]
                A += max_item_mean[:,None].dot(max_item_mean[None,:])
                num_user_candidate_items -= 1
                if self.get_reward(uid,max_i) >= self.threshold:
                    b += self.get_reward(uid,max_i)*max_item_mean
                    REC_ONE = True
                    num_correct_items += 1
                    if self.exit_when_consumed_all and num_correct_items == self.users_num_correct_items[uid]:
                        print(f"Exiting user {uid} with {len(result)} items in total and {num_correct_items} correct ones")
                        return np.array(result)

        return np.array(result)
