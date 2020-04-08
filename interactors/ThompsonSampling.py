from .ICF import ICF
import numpy as np
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
class ThompsonSampling(ICF):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def interact(self, uids, items_means,items_covs):
        super().interact()
        self.items_means = items_means
        self.items_covs = items_covs
        num_users = len(uids)
        # get number of latent factors 

        with threadpool_limits(limits=1, user_api='blas'):
            args = [(int(uid),) for uid in uids]
            result = util.run_parallel(self.interact_user,args)
        for i, user_result in enumerate(result):
            self.result[uids[i]] = user_result
        self.save_result()

    @classmethod
    def interact_user(cls, uid):
        self = cls.getInstance()
        num_lat = len(self.items_means[0])
        I = np.eye(num_lat)

        user_candidate_items = list(range(len(self.items_means)))
        # get number of latent factors 
        b = np.zeros(num_lat)
        A = self.user_lambda*I
        result = []
        for i in range(self.interactions):
            mean = np.dot(np.linalg.inv(A),b)
            cov = np.linalg.inv(A)*self.var
            p = np.random.multivariate_normal(mean,cov)
            max_i = np.NAN
            max_q = np.NAN
            max_e_reward = np.NINF
            for item in user_candidate_items:
                item_mean = self.items_means[item]
                item_cov = self.items_covs[item]
                q = np.random.multivariate_normal(item_mean,item_cov)
                e_reward = p @ q
                if e_reward > max_e_reward:
                    max_i = item
                    max_q = q
                    max_e_reward = e_reward
            user_candidate_items.remove(max_i)
            A += max_q[:,None].dot(max_q[None,:])
            b += self.get_reward(uid,max_i)*max_q
            result.append(max_i)
        return result
