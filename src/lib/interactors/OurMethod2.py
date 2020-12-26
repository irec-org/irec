import numpy as np
from tqdm import tqdm
import util
import interactors
from threadpoolctl import threadpool_limits
import ctypes
import scipy.spatial
import matplotlib.pyplot as plt
import os
import pickle
import sklearn
import scipy.optimize
class OurMethod2(interactors.Interactor):
    def __init__(self, alpha=1.0,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def interact(self, items_latent_factors):
        super().interact()
        uids = self.test_users
        num_lat = len(items_latent_factors[0])
        self.items_latent_factors = items_latent_factors
        num_users = len(uids)

        items_entropy = interactors.Entropy.get_items_entropy(self.train_consumption_matrix)
        items_popularity = interactors.MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)
        # self.items_bias = interactors.PPELPE.get_items_ppelpe(items_popularity,items_entropy)
        self.items_bias = interactors.LogPopEnt.get_items_logpopent(items_popularity,items_entropy)
        assert(self.items_bias.min() >= 0 and self.items_bias.max() == 1)

        # regression_model = sklearn.linear_model.LinearRegression()
        res=scipy.optimize.minimize(lambda x,items_latent_factors,items_bias: np.linalg.norm(items_bias - x @ items_latent_factors.T),
                                    np.ones(num_lat),
                                    args=(self.items_latent_factors,self.items_bias))
        self.initial_b = res.x 

        print(np.corrcoef(self.items_bias,self.initial_b @ items_latent_factors.T)[0,1])
        # users_bs = {}
        # for uid in uids:
        #     users_bs[uid] = np.ones(num_lat)
        #     users_bs[uid] = res.x
        self_id = id(self)
        np.seterr('warn')
        with threadpool_limits(limits=1, user_api='blas'):
            args = [(self_id,int(uid),) for uid in uids]
            results = util.run_parallel(self.interact_user,args)

        for i, user_result in enumerate(results):
            self.results[uids[i]] = user_result

        self.save_results()

    @staticmethod
    def interact_user(obj_id,uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        num_lat = len(self.items_latent_factors[0])
        I = np.eye(num_lat)

        user_candidate_items = np.array(list(range(len(self.items_latent_factors))))
        num_user_candidate_items = len(user_candidate_items)
        b = self.initial_b.copy()
        # regression_model = sklearn.linear_model.LinearRegression()
        # (
        #     self.items_latent_factors)
        A = I.copy()
        result = []
        for i in range(self.interactions):
            user_latent_factors = np.dot(np.linalg.inv(A),b)
            items_uncertainty = np.sqrt(np.sum(self.items_latent_factors[user_candidate_items].dot(np.linalg.inv(A)) * self.items_latent_factors[user_candidate_items],axis=1))
            items_user_similarity = user_latent_factors @ self.items_latent_factors[user_candidate_items].T
            user_model_items_score = items_user_similarity + self.alpha*items_uncertainty
            items_score = user_model_items_score

            best_items = user_candidate_items[np.argsort(items_score)[::-1]][:self.interaction_size]
            user_candidate_items = user_candidate_items[~np.isin(user_candidate_items,best_items)]
            result.extend(best_items)

            # num_correct_items_history.append(0)
            for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
                max_item_latent_factors = self.items_latent_factors[max_i]
                A += max_item_latent_factors[:,None].dot(max_item_latent_factors[None,:])
                num_user_candidate_items -= 1
                if self.get_reward(uid,max_i) >= self.threshold:
                    b += self.get_reward(uid,max_i)*max_item_latent_factors
                    if self.exit_when_consumed_all and num_correct_items == self.users_num_correct_items[uid]:
                        print(f"Exiting user {uid} with {len(result)} items in total and {num_correct_items} correct ones")
                        return np.array(result)

                    
            # old_mean = mean.copy()

        return np.array(result)
