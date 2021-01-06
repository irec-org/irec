import numpy as np
from tqdm import tqdm
from .ExperimentalInteractor import ExperimentalInteractor
import matplotlib.pyplot as plt
import os
import scipy.sparse
from collections import defaultdict
import random

def _softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
class PTS(ExperimentalInteractor):
    def __init__(self,num_particles,self.var,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_particles = num_particles
        self.var = var

    def interact(self,items_means):
        super().interact()
        uids = self.test_users
        num_items = self.train_consumption_matrix.shape[1]
        num_lat = len(self.items_means[0])
        particles = [{'u':np.random.normal(len(uids),num_lat),'v':items_means,'var_u':1.0, 'var_i':1.0} for i in range(self.num_particles)]

        particles_ids = np.arange(self.num_particles)

        # users_history = scipy.sparse.csr(self.train_consumption.shape)
        # items_popularity = MostPopular.get_items_popularity(self.train_consumption_matrix, normalize=False)

        # top_iids = list(reversed(np.argsort(items_popularity)))[:self.get_iterations()]
        num_users = len(uids)
        users_num_interactions = defaultdict(int)
        available_users = set(uids)
        item_users_consumed = defaultdict(list) 
        for i in tqdm(range(num_users*self.interactions)):
            uid = random.sample(available_users,k=1)[0]
            not_recommended = np.ones(num_items,dtype=bool)
            not_recommended[self.results[uid]] = 0
            items_not_recommended = np.nonzero(not_recommended)[0]

            particle_idx = np.random.choice(particles_ids)
            particle = particles[particle_idx]
            items_score = particle['u'] @ particle['v'][items_not_recommended].T
            best_items = items_not_recommended[np.argsort(items_score)[::-1]][:self.interaction_size]

            
            for best_item in best_items:
                # users_history[uid,best_item] = self.get_reward(uid,best_item)
                if self.get_reward(uid,best_item) >= self.threshold:
                    lambdas_u_i = []
                    etas_u_i  = []
                    mus_u_i = []
                    v_j = particle['v'][self.results[uid]]
                    for particle in particles:
                        lambda_u_i = 1/self.var*(v_j.T @ v_j)+1/particle['var_u'] * np.eye(num_lat)
                        eta_u_i = np.sum(np.array([self.get_reward(uid,result) for result in self.results[uid]]) * v_j)
                        reward = self.get_reward(uid,best_item)
                        lambdas_u_i.append(lambda_u_i)
                        etas_u_i.append(eta_u_i)
                        mus_u_i.append(1/self.var*(np.linalg.inv(lambda_u_i) @ eta_u_i))

                    weights = []
                    for particle, lambda_u_i, mu_u_i in zip(particles, mus_u_i, lambdas_u_i):
                        v_j = particle['v'][self.results[uid]]
                        cov = 1/self.var + v_j.T @ mu_u_i @ v_j
                        w = np.random.normal(
                            v_j.T @ mu,
                            cov
                        )
                        weights.append(w)

                    normalized_weights = _softmax(weights)
                    ds = [np.random.choice(range(self.num_particles), p=normalized_weights) for _ in range(self.num_particles)]
                    new_particles = [{"u": np.copy(particles[d]["u"]),
                      "v": np.copy(particles[d]["v"]),
                      "var_u": particles[d]["var_u"],
                      "var_i": particles[d]["var_i"]} for d in ds]
                    for idx, (particle, lambda_u_i, eta_u_i) in enumerate(zip(new_particles, lambdas_u_i, etas_u_i)):
                        v_j = particle["v"][best_item, :]
                        lambda_u_i += 1/self.var * (v_j @ v_j.T)
                        eta_u_i += reward * v_j
                        inv_lambda_u_i = np.linalg.inv(lambda_u_i)
                        sampled_user_vector = np.random.multivariate_normal(1/self.var*(inv_lambda_u_i @ eta_u_i), inv_lambda_u_i)
                        new_particles[idx]['u'][uid] = sampled_user_vector

                        u_i = particle["u"][item_users_consumed[best_item]]
                        lambda_v_i = 1/self.var * (u_i.T @ u_i) + 1/particle['var_i']*np.eye(num_lat)

                        eta = np.sum(u_i * np.array([self.get_reward(uid,best_item) in item_users_consumed]))
                        inv_lambda_v_i = np.linalg.inv(lambda_v_i)
                        item_sample_vector = np.random.multivariate_normal(1/self.var*(inv_lambda_v_i @ eta-u_i),inv_lambda_v_i)

                        new_particles[idx]['v'][best_item] = item_sample_vector
                    
                        
            self.results[uid].extend(best_items)

            for best_item in best_items:
                item_users_consumed[uid].extend(best_items)
            users_num_interactions[uid] += 1
            if users_num_interactions[uid] == self.interactions:
                available_users = available_users - {uid}

        self.save_results()
