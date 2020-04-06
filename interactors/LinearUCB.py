from .ICF import ICF
import numpy as np
from tqdm import tqdm
import util
class LinearUCB(ICF):
    def __init__(self, alpha=0.2, zeta=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha != None:
            self.alpha = alpha
        elif zeta != None:
            self.alpha = 1+np.sqrt(np.log(2/zeta)/2)

    def interact(self, uids, items_means):
        super().interact()
        self.items_means = items_means
        num_users = len(uids)
        # get number of latent factors 
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

        u_items_means = self.items_means.copy()
        b = np.zeros(num_lat)
        A = self.user_lambda*I
        result = []
        for i in range(self.interactions):
            mean = np.dot(np.linalg.inv(A),b)
            cov = np.linalg.inv(A)*self.var
            max_i = np.NAN
            max_item_mean = np.NAN
            max_reward = np.NINF
            for item, item_mean in zip(u_items_means.keys(),u_items_means.values()):
                # q = np.random.multivariate_normal(item_mean,item_cov)
                reward = mean.T @ item_mean + self.alpha*np.sqrt(item_mean.T.dot(cov).dot(item_mean))
                if reward > max_reward:
                    max_i = item
                    max_item_mean = item_mean
                    max_reward = reward
            del u_items_means[max_i]

            A += max_item_mean.dot(max_item_mean.T)
            b += self.get_reward(uid,max_i)*max_item_mean
            result.append(max_i)
        return result
