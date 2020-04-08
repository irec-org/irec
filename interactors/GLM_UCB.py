from .ICF import ICF
import numpy as np
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
class GLM_UCB(ICF):
    def __init__(self, c=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def p(self,x):
        return self.sigmoid(x)

    def interact(self, uids, items_means):
        super().interact()
        self.items_means = items_means
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
        b = np.zeros(num_lat)
        p = np.zeros(num_lat)
        u_rec_rewards = []
        u_rec_items_means = []
        A = self.user_lambda*I
        result = []
        for i in range(self.interactions):
            for j in range(self.interaction_size):
                if len(u_rec_items_means) == 0:
                    p = np.zeros(num_lat)
                else:
                    p = np.sum(np.array(
                        [(u_rec_rewards[t] - self.p(p.T @ u_rec_items_means[t]))*u_rec_items_means[t]
                            for t in range(0,len(u_rec_items_means))]
                    ),axis=0)
                cov = np.linalg.inv(A)*self.var
                max_i = np.NAN
                max_item_mean = np.NAN
                max_e_reward = np.NINF
                for item in user_candidate_items:
                    item_mean = self.items_means[item]
                    # q = np.random.multivariate_normal(item_mean,item_cov)
                    e_reward = self.p(p.T @ item_mean) + self.c * np.sqrt(np.log(i+1)) * np.sqrt(item_mean.T.dot(cov).dot(item_mean))
                    if e_reward > max_e_reward:
                        max_i = item
                        max_item_mean = item_mean
                        max_e_reward = e_reward
                user_candidate_items.remove(max_i)
                result.append(max_i)
                if self.get_reward(uid,max_i) >= self.values[-2]:
                    u_rec_rewards.append(self.get_reward(uid,max_i))
                    u_rec_items_means.append(max_item_mean)
                    A += max_item_mean[:,None].dot(max_item_mean[None,:])
        return result
