from .Interactor import Interactor
import numpy as np
from tqdm import tqdm
import util
from threadpoolctl import threadpool_limits
import ctypes
class LinUCB(Interactor):
    def __init__(self, alpha=1.0, zeta=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha != None:
            self.alpha = alpha
        elif zeta != None:
            self.alpha = 1+np.sqrt(np.log(2/zeta)/2)

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[2],(self.train_dataset.data[0],self.train_dataset.data[1])),(self.train_dataset.users_num,self.train_dataset.items_num))
        self.num_items = self.train_dataset.num_items

        mf_model = mf.SVD()
        mf_model.fit(self.train_consumption_matrix)
        self.items_weights = mf_model.items_weights

        self.I = np.eye(len(self.items_weights[0]))
        bs = defaultdict(lambda: np.zeros(self.num_latent_factors))
        As = defaultdict(lambda: self.I.copy())

    def predict(self,uid,candidate_items,num_req_items):
        b = bs[uid]
        A = As[uid]
        mean = np.dot(np.linalg.inv(A),b)
        items_uncertainty = self.alpha*np.sqrt(np.sum(self.items_weights[candidate_items].dot(np.linalg.inv(A)) * self.items_weights[candidate_items],axis=1))
        items_user_similarity = mean @ self.items_weights[candidate_items].T
        items_score =  items_user_similarity + items_uncertainty
        return items_score, None

    def update(self,uid,item,reward,additional_data):
        max_item_latent_factors = self.items_weights[item]
        b = bs[uid]
        A = As[uid]
        A += max_item_latent_factors[:,None].dot(max_item_latent_factors[None,:])
        b += reward*max_item_latent_factors

