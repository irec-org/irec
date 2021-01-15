import numpy as np
from tqdm import tqdm
from .ExperimentalInteractor import ExperimentalInteractor
import matplotlib.pyplot as plt
import os
import scipy.sparse
from collections import defaultdict
import random
from .MFInteractor import MFInteractor

def _softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
class PTS(MFInteractor):
    def __init__(self,num_particles,var, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_particles = num_particles
        self.var = var
        self.parameters.extend(['num_particles','var'])

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[:,2],(self.train_dataset.data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.num_total_users,self.train_dataset.num_total_items))

        self.consumption_matrix = self.train_consumption_matrix.tolil()

        self.num_total_items = self.train_dataset.num_total_items
        self.num_total_users = self.train_dataset.num_total_users

        self.particles = [{'u':np.random.normal(size=(self.num_total_users,self.num_lat)),
                           'v': np.random.normal(size=(self.num_total_items, self.num_lat)),
                           'var_u':1.0,
                           'var_i':1.0}
                          for i in range(self.num_particles)]

        self.particles_ids = np.arange(self.num_particles)
        self.items_consumed_users = defaultdict(list)
        self.items_consumed_users_ratings = defaultdict(list)
        self.users_consumed_items = defaultdict(list)
        self.users_consumed_items_ratings = defaultdict(list)
        for row in self.train_dataset.data:
            uid = int(row[0])
            iid = int(row[1])
            reward = row[2]
            self.update(uid,iid,reward,None)
        
    def predict(self,uid,candidate_items,num_req_items):

        particle_idx = np.random.choice(self.particles_ids)
        particle = self.particles[particle_idx]
        items_score = particle['u'][uid] @ particle['v'][candidate_items].T

        return items_score, None

    def update(self,uid,item,reward,additional_data):
        best_item = item
        if reward >= self.train_dataset.mean_rating:
            lambdas_u_i = []
            zetas_u_i  = []
            mus_u_i = []
            for particle in self.particles:
                v_j = particle['v'][self.users_consumed_items[uid]]
                lambda_u_i = 1/self.var*(v_j.T @ v_j)+1/particle['var_u'] * np.eye(self.num_lat)
                zeta_u_i = np.sum(np.multiply(v_j,np.array(self.users_consumed_items_ratings[uid]).reshape(-1, 1)),axis=0)
                # reward = self.get_reward(uid,best_item)
                lambdas_u_i.append(lambda_u_i)
                zetas_u_i.append(zeta_u_i)
                mus_u_i.append(1/self.var*(np.linalg.inv(lambda_u_i) @ zeta_u_i))

            weights = []
            for particle, lambda_u_i, mu_u_i in zip(self.particles, lambdas_u_i, mus_u_i):
                v_j = particle['v'][self.users_consumed_items[uid]]
                cov = 1/self.var + v_j.T @ mu_u_i @ v_j
                w = np.random.normal(v_j.T @ mu,cov)
                weights.append(w)

            normalized_weights = _softmax(weights)
            ds = np.random.choice(range(self.num_particles), p=normalized_weights,size=self.num_particles)
            new_particles = [{"u": np.copy(self.particles[d]["u"]),
                "v": np.copy(self.particles[d]["v"]),
                "var_u": self.particles[d]["var_u"],
                "var_i": self.particles[d]["var_i"]} for d in ds]
            for idx, (particle, lambda_u_i, zeta_u_i) in enumerate(zip(new_particles, lambdas_u_i, zetas_u_i)):
                v_j = particle["v"][best_item, :]
                lambda_u_i += 1/self.var * (v_j @ v_j.T)
                zeta_u_i += reward * v_j
                inv_lambda_u_i = np.linalg.inv(lambda_u_i)
                sampled_user_vector = np.random.multivariate_normal(1/self.var*(inv_lambda_u_i @ zeta_u_i), inv_lambda_u_i)
                new_particles[idx]['u'][uid] = sampled_user_vector

                u_i = particle["u"][self.items_consumed_users[best_item]]
                lambda_v_i = 1/self.var * (u_i.T @ u_i) + 1/particle['var_i']*np.eye(self.num_lat)

                zeta = np.sum(u_i * np.array([self.consumption_matrix[uid,best_item] in self.items_consumed_users]))
                inv_lambda_v_i = np.linalg.inv(lambda_v_i)
                item_sample_vector = np.random.multivariate_normal(1/self.var*(inv_lambda_v_i @ zeta-u_i),inv_lambda_v_i)

                new_particles[idx]['v'][best_item] = item_sample_vector
            self.particles = new_particles

            self.users_consumed_items[uid].append(item)
            self.users_consumed_items_ratings[uid].append(reward)
            self.items_consumed_users[iid].append(uid)
            self.items_consumed_users_ratings[iid].append(reward)

            self.consumption_matrix[uid,item] = reward
