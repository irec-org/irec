from .ICF import ICF
import numpy as np
from tqdm import tqdm
class ThompsonSampling(ICF):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def interact(self, uids, items_means,items_covs):
        super().interact()
        num_users = len(uids)
        num_lat = len(items_means[0])
        I = np.eye(num_lat)
        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            u_items_means = items_means.copy()
            u_items_covs = items_covs.copy()
            # get number of latent factors 
            b = np.zeros(num_lat)
            A = self.user_lambda*I
            for i in range(self.interactions):
                mean = np.dot(np.linalg.inv(A),b)
                cov = np.linalg.inv(A)*self.var
                p = np.random.multivariate_normal(mean,cov)
                max_i = np.NAN
                max_q = np.NAN
                max_reward = np.NINF
                for item, (item_mean, item_cov) in zip(u_items_means.keys(),zip(u_items_means.values(), u_items_covs.values())):
                    q = np.random.multivariate_normal(item_mean,item_cov)
                    reward = p @ q
                    if reward > max_reward:
                        max_i = item
                        max_q = q
                        max_reward = reward
                del u_items_means[max_i]
                del u_items_covs[max_i]
                A += max_q.dot(max_q.T)
                b += self.get_reward(uid,max_i)*max_q
                self.result[uid].append(max_i)
        self.save_result()
