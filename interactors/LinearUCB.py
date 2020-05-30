from .ICF import ICF
import numpy as np
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
import ctypes
class LinearUCB(ICF):
    def __init__(self, alpha=1.0, zeta=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha != None:
            self.alpha = alpha
        elif zeta != None:
            self.alpha = 1+np.sqrt(np.log(2/zeta)/2)

    def interact(self, items_means):
        super().interact()
        uids = self.test_users
        self.items_means = items_means
        num_users = len(uids)
        # get number of latent factors 

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

    @staticmethod
    def interact_user(obj_id,uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        if not issubclass(self.__class__,ICF): # DANGER CODE
            raise RuntimeError
        num_lat = len(self.items_means[0])
        I = np.eye(num_lat)

        user_candidate_items = np.array(list(range(len(self.items_means))))
        b = np.zeros(num_lat)
        A = self.user_lambda*I
        result = []

        num_correct_items = 0

        for i in range(self.interactions):
            mean = np.dot(np.linalg.inv(A),b)
            cov = np.linalg.inv(A)*self.var

            best_items  = user_candidate_items[np.argsort(mean @ self.items_means[user_candidate_items].T+\
                                                          self.alpha*np.sqrt(np.sum(self.items_means[user_candidate_items].dot(cov) * self.items_means[user_candidate_items],axis=1)))[::-1]][:self.interaction_size]

            user_candidate_items = user_candidate_items[~np.isin(user_candidate_items,best_items)]
            result.extend(best_items)

            # if i == 0:
            #     if uid >= 0 and uid <= 300:
            #         print('user',uid,'first top-5',result[i*self.interaction_size:(i+1)*self.interaction_size])

            for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
                max_item_mean = self.items_means[max_i]
                A += max_item_mean[:,None].dot(max_item_mean[None,:])
                if self.get_reward(uid,max_i) >= self.threshold:
                    b += self.get_reward(uid,max_i)*max_item_mean
                    num_correct_items += 1
                    if self.exit_when_consumed_all and num_correct_items == self.users_num_correct_items[uid]:
                        print(f"Exiting user {uid} with {len(result)} items in total and {num_correct_items} correct ones")
                        return np.array(result)

        return np.array(result)
