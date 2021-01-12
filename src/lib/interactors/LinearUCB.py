from .ICF import ICF
import numpy as np
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import ctypes
from collections import defaultdict

class LinearUCB(ICF):
    def __init__(self, alpha=1.0, zeta=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha != None:
            self.alpha = alpha
        elif zeta != None:
            self.alpha = 1+np.sqrt(np.log(2/zeta)/2)

        self.parameters.extend(['alpha'])

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[:,2],(self.train_dataset.data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.users_num,self.train_dataset.items_num))
        self.num_items = self.train_dataset.num_items
        mf_model = mf.ICFPMFS()
        mf_model.fit(self.train_consumption_matrix)
        self.items_means = mf_model.items_means

        self.num_latent_factors = len(self.items_latent_factors[0])

        self.I = np.eye(self.num_latent_factors)
        bs = defaultdict(lambda: np.zeros(self.num_latent_factors))
        As = defaultdict(lambda: self.user_lambda*I)

    def predict(self,uid,candidate_items,num_req_items):
        b = bs[uid]
        A = As[uid]
        mean = np.dot(np.linalg.inv(A),b)
        cov = np.linalg.inv(A)*self.var
        
        items_score  = mean @ self.items_means[candidate_items].T+\
            self.alpha*np.sqrt(np.sum(self.items_means[candidate_items].dot(cov) * self.items_means[candidate_items],axis=1))
        
        return items_score, None

    def update(self,uid,item,reward,additional_data):
        max_item_mean = self.items_means[item]
        A += max_item_mean[:,None].dot(max_item_mean[None,:])
        b += reward*max_item_mean
