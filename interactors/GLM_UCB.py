from .ICF import ICF
import numpy as np
from tqdm import tqdm
class GLM_UCB(ICF):
    def __init__(self, c=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def p(self,x):
        return x

    def interact(self, uids, items_means):
        super().interact()
        num_users = len(uids)
        # get number of latent factors 
        num_lat = len(items_means[0])

        I = np.eye(num_lat)
        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            u_items_means = items_means.copy()
            b = np.zeros(num_lat)
            p = np.zeros(num_lat)
            u_rec_rewards = []
            u_rec_items_means = []
            A = self.user_lambda*I
            for i in range(self.interactions):
                if i == 0:
                    p = np.zeros(num_lat)
                else:
                    p = np.sum(np.array(
                        [(u_rec_rewards[t] - self.p(p.T @ u_rec_items_means[t]))*u_rec_items_means[t]
                         for t in range(0,i)]
                    ),axis=0)
                cov = np.linalg.inv(A)*self.var
                max_i = np.NAN
                max_item_mean = np.NAN
                max_reward = np.NINF
                for item, item_mean in zip(u_items_means.keys(),u_items_means.values()):
                    # q = np.random.multivariate_normal(item_mean,item_cov)
                    e_reward = self.p(p.T @ item_mean) + self.c * np.sqrt(np.log(i+1)) * np.sqrt(item_mean.T.dot(cov).dot(item_mean))
                    if e_reward > max_reward:
                        max_i = item
                        max_item_mean = item_mean
                        max_reward = self.get_reward(uid,max_i)
                del u_items_means[max_i]

                u_rec_rewards.append(max_reward)
                u_rec_items_means.append(max_item_mean)
                A += max_item_mean.dot(max_item_mean.T)
                self.result[uid].append(max_i)
        self.save_result()

