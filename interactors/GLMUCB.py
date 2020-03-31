class GLMUCB(ICF):
    def __init__(self, c=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def p(self,x):
        sigmoid(x)

    def interact(self, uids, items_means):
        num_users = len(uids)
        already_computed = 0
        # get number of latent factors 
        num_lat = len(items_means[0][0])

        I = np.eye(num_lat)

        for uid in uids:
            u_items_means = items_means.copy()
            print(f'[{already_computed}/{num_users}]')
            b = np.zeros(num_lat)
            p = np.zeros(num_lat)
            u_rec_rewards = []
            u_rec_items_means = []
            time = 1
            A = self.u_lambda*I
            for i in range(self.interactions):
                for j in range(self.interaction_size):
                    p = np.sum(np.array([(u_rec_rewards[t] - self.p(p.T @ u_rec_items_means[t]))*u_rec_items_means[t] for t in range(1,time)]),axis=0)
                    cov = np.linalg.inv(A)*self.var
                    max_i = np.NAN
                    max_item_mean = np.NAN
                    max_reward = np.NINF
                    for item, item_mean in zip(u_items_means.keys(),u_items_means):
                        # q = np.random.multivariate_normal(item_mean,item_cov)
                        e_reward = self.p(p.T @ item_mean) + self.c * np.sqrt(np.log(t)) * np.sqrt(item_mean.T.dot(cov).dot(item_mean))
                        if e_reward > max_reward:
                            max_i = item
                            max_item_mean = item_mean
                            max_reward = self.get_reward(uid,max_i)
                    del u_items_means[max_i]

                    u_rec_rewards.append(max_reward)
                    A += max_item_mean.dot(max_item_mean.T)
                    time += 1
            already_computed += 1
