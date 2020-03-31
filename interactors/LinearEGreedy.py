class LinearEGreedy(ICF):
    def __init__(self, epsilon=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

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
            A = self.u_lambda*I
            for i in range(self.interactions):
                for j in range(self.interaction_size):
                    mean = np.dot(np.linalg.inv(A),b)
                    max_i = np.NAN
                    max_item_mean = np.NAN
                    max_reward = np.NINF
                    if self.epsilon < np.random.rand():
                        for item, item_mean in zip(u_items_means.keys(),u_items_means):
                            # q = np.random.multivariate_normal(item_mean,item_cov)
                            reward = mean.T @ item_mean
                            if reward > max_reward:
                                max_i = item
                                max_item_mean = item_mean
                                max_reward = reward
                    else:
                        max_i = random.choice(u_items_means.keys())
                        max_item_mean = u_items_means[max_i]
                        max_reward = mean.T @ max_item_mean
                    del u_items_means[max_i]
                    A += max_item_mean.dot(max_item_mean.T)
                    b += self.get_reward(uid,max_i)*max_item_mean
            already_computed += 1
