from threadpoolctl import threadpool_limits
import numpy as np
from tqdm import tqdm
from .ExperimentalInteractor import ExperimentalInteractor
import matplotlib.pyplot as plt
import os
import scipy.sparse
from collections import defaultdict
import random
from .MFInteractor import MFInteractor
from tqdm import tqdm
from numba import njit, jit

@njit
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

        self.num_total_items = self.train_dataset.num_total_items
        self.num_total_users = self.train_dataset.num_total_users

        self.particles_us = np.random.normal(size=(self.num_particles,self.num_total_users,self.num_lat))
        self.particles_vs = np.random.normal(size=(self.num_particles,self.num_total_items,self.num_lat))
        self.particles_var_us = np.ones(shape=(self.num_particles))
        self.particles_var_is = np.ones(shape=(self.num_particles))

        self.particles_ids = np.arange(self.num_particles)
        self.items_consumed_users = defaultdict(list)
        self.items_consumed_users_rewards = defaultdict(list)
        self.users_consumed_items = defaultdict(list)
        self.users_consumed_items_rewards = defaultdict(list)
        for i in tqdm(range(len(self.train_dataset.data))):
            uid = int(self.train_dataset.data[i,0])
            item = int(self.train_dataset.data[i,1])
            reward = self.train_dataset.data[i,2]
            self.update(uid,item,reward,None)
        
    def predict(self,uid,candidate_items,num_req_items):
        particle_idx = np.random.choice(self.particles_ids)
        items_score = self.particles_us[particle_idx][uid] @ self.particles_vs[particle_idx][candidate_items].T
        return items_score, None

    def update(self,uid,item,reward,additional_data):
        with threadpool_limits(limits=1, user_api='blas'):
            updated_history = False
            lambdas_u_i = np.empty(shape=(self.num_particles,self.num_lat,self.num_lat))
            zetas_u_i  = np.empty(shape=(self.num_particles,self.num_lat))
            mus_u_i = np.empty(shape=(self.num_particles,self.num_lat))
            for i in range(self.num_particles):
                v_j = self.particles_vs[i][self.users_consumed_items[uid]]
                lambda_u_i = 1/self.var*(v_j.T @ v_j)+1/self.particles_var_us[i] * np.eye(self.num_lat)
                zeta_u_i = np.sum(np.multiply(v_j,np.array(self.users_consumed_items_rewards[uid]).reshape(-1, 1)),axis=0)
                lambdas_u_i[i]= lambda_u_i
                zetas_u_i[i] = zeta_u_i
                mus_u_i[i] = 1/self.var*(np.linalg.inv(lambda_u_i) @ zeta_u_i)

            weights = np.empty(self.num_particles)
            for i in range(self.num_particles):
                lambda_u_i, mu_u_i = lambdas_u_i[i], mus_u_i[i]
                v_j = self.particles_vs[i][item,:]
                cov = 1/self.var + np.dot(np.dot(v_j.T, lambda_u_i), v_j)
                w = scipy.stats.norm(np.dot(v_j.T, mu_u_i),cov).pdf(reward)
                weights[i]=w

            normalized_weights = _softmax(weights)
            ds = np.random.choice(range(self.num_particles), p=normalized_weights,size=self.num_particles)
            new_particles_us = np.empty(shape=(self.num_particles,self.num_total_users,self.num_lat))
            new_particles_vs = np.empty(shape=(self.num_particles,self.num_total_items,self.num_lat))
            new_particles_var_us = np.empty(shape=(self.num_particles))
            new_particles_var_is = np.empty(shape=(self.num_particles))

            for i in range(self.num_particles):
                d = ds[i]
                new_particles_us[i]=self.particles_us[d]
                new_particles_vs[i]=self.particles_vs[d]
                new_particles_var_us[i]=self.particles_var_us[d]
                new_particles_var_is[i]=self.particles_var_is[d]

            if not updated_history:
                self.users_consumed_items[uid].append(item)
                self.users_consumed_items_rewards[uid].append(reward)
                self.items_consumed_users[item].append(uid)
                self.items_consumed_users_rewards[item].append(reward)
                updated_history = True

            for i in range(self.num_particles):
                lambda_u_i, zeta_u_i = lambdas_u_i[i], zetas_u_i[i]
                v_j = new_particles_vs[i][item, :]
                lambda_u_i += 1/self.var * (v_j @ v_j.T)
                zeta_u_i += reward * v_j

                inv_lambda_u_i = np.linalg.inv(lambda_u_i)
                sampled_user_vector = np.random.multivariate_normal(1/self.var*(inv_lambda_u_i @ zeta_u_i), inv_lambda_u_i)
                new_particles_us[i][uid] = sampled_user_vector

                u_i = new_particles_us[i][self.items_consumed_users[item],:]
                lambda_v_i = 1/self.var * (u_i.T @ u_i) + 1/new_particles_var_is[i]*np.eye(self.num_lat)

                zeta = np.sum(np.multiply(u_i,np.array(self.items_consumed_users_rewards[item]).reshape(-1, 1)),axis=0)
                inv_lambda_v_i = np.linalg.inv(lambda_v_i)
                item_sample_vector = np.random.multivariate_normal(1/self.var*(inv_lambda_v_i @ zeta),inv_lambda_v_i)
                new_particles_vs[i][item] = item_sample_vector

            if not updated_history:
                self.users_consumed_items[uid].append(item)
                self.users_consumed_items_rewards[uid].append(reward)
                self.items_consumed_users[item].append(uid)
                self.items_consumed_users_rewards[item].append(reward)
                updated_history = True

            self.particles_us=new_particles_us
            self.particles_vs=new_particles_vs
            self.particles_var_us=new_particles_var_us
            self.particles_var_is=new_particles_var_is
