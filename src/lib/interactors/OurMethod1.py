import numpy as np
from tqdm import tqdm
import util
import interactors
from threadpoolctl import threadpool_limits
import ctypes
import scipy.spatial
import matplotlib.pyplot as plt
import os
import pickle
class OurMethod1(interactors.ExperimentalInteractor):
    def __init__(self, alpha=1.0, stop=None, weight_method='change',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters['alpha'] = alpha
        self.parameters['weight_method'] = weight_method
        self.parameters['stop'] = stop

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[2],(self.train_dataset.data[0],self.train_dataset.data[1])),(self.train_dataset.users_num,self.train_dataset.items_num))
        self.num_items = self.train_dataset.num_items

        mf_model = mf.SVD()
        mf_model.fit(self.train_consumption_matrix)
        self.items_weights = mf_model.items_weights


        # self.items_weights = items_weights
        # num_users = len(uids)

        items_entropy = interactors.Entropy.get_items_entropy(self.train_consumption_matrix)
        items_popularity = interactors.MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)
        # self.items_bias = interactors.PPELPE.get_items_ppelpe(items_popularity,items_entropy)
        self.items_bias = interactors.LogPopEnt.get_items_logpopent(items_popularity,items_entropy)

        assert(self.items_bias.min() >= 0 and self.items_bias.max() == 1)

        self.I = np.eye(len(self.items_weights[0]))
        bs = defaultdict(lambda: np.zeros(self.num_latent_factors))
        As = defaultdict(lambda: self.I.copy())

        users_num_correct_items_history= defaultdict(lambda:[0])
        users_num_correct_items_history = defaultdict(lambda:[0])
        users_similarity_score = defaultdict(lambda:[0])
        users_distance_history = defaultdict(lambda:[])
        users_global_model_weights = defaultdict(lambda:[])

        # user_latent_factors_history = np.empty(shape=(0, num_lat))
        # num_correct_items_history = [0]
        # similarity_score = [0]
        # distance_history = []
        # global_model_weights = []



    def get_global_model_weight(self,user_latent_factors_history,num_correct_items_history,distance_history):
        if self.parameters['weight_method'] == 'stop':
            b = self.parameters['stop']
            a = min(np.sum(num_correct_items_history),b)
            return (1-np.round(pow(2,a)/pow(2,b),3))
        elif self.parameters['weight_method'] == 'change':
            if len(user_latent_factors_history) == 0:
                return 1
            times_with_reward = np.nonzero(num_correct_items_history)[0]
            if len(times_with_reward) < 2:
                return 1
            # print(user_latent_factors_history[times_with_reward][-1])
            # print(user_latent_factors_history[times_with_reward][-2])
            res = scipy.spatial.distance.cosine(user_latent_factors_history[times_with_reward][-1],user_latent_factors_history[times_with_reward][-2])
            distance_history.append(res)
            res = res/np.max(distance_history)
            # res = (res+1)/2
            return res
        else:
            raise RuntimeError

    def predict(self,uid,candidate_items,num_req_items):
        b = bs[uid]
        A = As[uid]
        
        user_latent_factors_history = users_num_correct_items_history[uid]
        num_correct_items_history   = users_num_correct_items_history[uid]
        similarity_score            = users_similarity_score[uid]
        distance_history            = users_distance_history[uid]
        global_model_weights        = users_global_model_weights[uid]

        user_latent_factors = np.dot(np.linalg.inv(A),b)
        user_latent_factors_history = np.vstack([user_latent_factors_history,user_latent_factors])
        global_model_weight = self.get_global_model_weight(user_latent_factors_history,
                                                            num_correct_items_history,
                                                            distance_history)
        global_model_weights.append(global_model_weight)
        items_uncertainty = np.sqrt(np.sum(self.items_weights[candidate_items].dot(np.linalg.inv(A)) * self.items_weights[candidate_items],axis=1))
        items_user_similarity = user_latent_factors @ self.items_weights[candidate_items].T
        user_model_items_score = items_user_similarity + self.parameters['alpha']*items_uncertainty
        global_model_items_score = self.items_bias[candidate_items]
        user_model_items_score_min = np.min(user_model_items_score)
        user_model_items_score_max = np.max(user_model_items_score)
        if user_model_items_score_max-user_model_items_score_min != 0:
            global_model_items_score = global_model_items_score*(user_model_items_score_max-user_model_items_score_min) + user_model_items_score_min

        items_score =  (1-global_model_weight)*user_model_items_score + global_model_weight*global_model_items_score
        return items_score, None


    def update(self,uid,item,reward,additional_data):
        max_item_latent_factors = self.items_weights[item]
        b = bs[uid]
        A = As[uid]
        A += max_item_latent_factors[:,None].dot(max_item_latent_factors[None,:])
        b += reward*max_item_latent_factors
        if reward > min(self.train_dataset.rate_domain):
            num_correct_items_history = users_num_correct_items_history[uid]
            num_correct_items_history[-1] += 1
