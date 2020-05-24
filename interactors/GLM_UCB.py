from .ICF import ICF
import numpy as np
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
import scipy.optimize
import ctypes
class GLM_UCB(ICF):
    def __init__(self, c=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def p(self,x):
        return x

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
            self.results[uids[i]] = user_result
        self.save_results()

    def error_user_weight_function(self,p,u_rec_rewards,u_rec_items_means):
        return np.sum(np.array(
            [(u_rec_rewards[t] - self.p(p.T @ u_rec_items_means[t]))*u_rec_items_means[t]
             for t in range(0,len(u_rec_items_means))]),0)

    @staticmethod
    def interact_user(obj_id,uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        if not issubclass(self.__class__,ICF): # DANGER CODE
            raise RuntimeError
        num_lat = len(self.items_means[0])
        I = np.eye(num_lat)

        user_candidate_items = list(range(len(self.items_means)))
        u_rec_rewards = []
        u_rec_items_means = []
        A = self.user_lambda*I
        result = []
        for i in range(self.interactions):
            if len(u_rec_items_means) == 0:
                p = np.zeros(num_lat)
            else:
                p = scipy.optimize.root(self.error_user_weight_function,
                                        p,
                                        (u_rec_rewards,u_rec_items_means)).x
            cov = np.linalg.inv(A)*self.var
            for j in range(self.interaction_size):
                max_i = user_candidate_items[np.argmax(self.p(p[None,:] @ self.items_means[user_candidate_items].T) + self.c * np.sqrt(np.log(i*self.interaction_size+j+1)) *\
                    np.sqrt(np.sum(self.items_means[user_candidate_items].dot(cov) * self.items_means[user_candidate_items],axis=1)))]

                user_candidate_items.remove(max_i)
                result.append(max_i)

            for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
                max_item_mean = self.items_means[max_i]
                u_rec_rewards.append(self.get_reward(uid,max_i))
                u_rec_items_means.append(max_item_mean)
                A += max_item_mean[:,None].dot(max_item_mean[None,:])
        return result
