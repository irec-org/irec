class ThompsonSampling(ICF):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def interact(self, uids, items_means,items_covs):
        num_users = len(uids)
        already_computed = 0
        I = np.eye(num_lat)
        for uid in uids:
            u_items_means = items_means.copy()
            u_items_covs = items_covs.copy()
            print(f'[{already_computed}/{num_users}]')
            # get number of latent factors 
            num_lat = len(u_items_means[0])
            b = np.zeros(num_lat)
            A = self.u_lambda*I
            for i in range(self.interactions):
                for j in range(self.interaction_size):
                    mean = np.dot(np.linalg.inv(A),b)
                    cov = np.linalg.inv(A)*self.var
                    p = np.random.multivariate_normal(mean,cov)
                    max_i = np.NAN
                    max_q = np.NAN
                    max_reward = np.NINF
                    for item, (item_mean, item_cov) in zip(u_items_means.keys(),zip(u_items_means, u_items_covs)):
                        q = np.random.multivariate_normal(item_mean,item_cov)
                        reward = p.dot(q)
                        if reward > max_reward:
                            max_i = item
                            max_q = q
                            max_reward = reward
                    del u_items_means[max_i]
                    del u_items_covs[max_i]
                    A += max_q.dot(max_q.T)
                    b += self.get_reward(uid,max_i)*max_q
            already_computed += 1
