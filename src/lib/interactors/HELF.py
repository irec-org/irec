import numpy as np
from tqdm import tqdm
from . import Interactor, Entropy, MostPopular,LogPopEnt
import matplotlib.pyplot as plt
import scipy.stats
import os

class HELF(ExperimentalInteractor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_helf(items_popularity,items_entropy,num_users):

        a = np.ma.log(items_popularity).filled(0)/np.log(num_users)
        b = items_entropy/np.max(items_entropy)
        print(np.sort(b))
        print(b.min())
        print(a.min())
        print(np.sort(a))
        np.seterr('warn')
        items_helf = 2*a*b/(a+b) 
        items_helf[np.isnan(items_helf)] = 0
        return items_helf

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[2],(self.train_dataset.data[0],self.train_dataset.data[1])),(self.train_dataset.users_num,self.train_dataset.items_num))
        self.num_items = self.train_dataset.num_items
        # mask = np.ones(self.train_consumption_matrix.shape[0], dtype=bool)
        # mask[uids] = 0
        num_train_users = len(self.train_dataset.uids)
        items_entropy = Entropy.get_items_entropy(self.train_consumption_matrix)
        items_popularity = MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)
        self.items_logpopent = HELF.get_items_helf(items_popularity,items_entropy,num_train_users)


    def predict(self,uid,candidate_items,num_req_items):
        items_score = self.items_logpopent[candidate_items]
        return items_score, None
        # top_items = list(reversed(np.argsort(items_score)))[:self.interaction_size]

        # top_iids = list(reversed(np.argsort(items_logpopent)))[:self.get_iterations()]
        # num_users = len(uids)


        # fig, ax = plt.subplots()
        # ax.scatter(items_entropy,items_popularity,marker="D",color='darkblue')
        # ax.set_ylabel("Popularity")
        # ax.set_xlabel("Entropy")
        # for start, end, color in [(0,10,'green'),(10,20,'red'),(20,30,'darkred'),(30,40,'yellow'),(40,50,'orange')]:
        #     ax.scatter(items_entropy[top_iids[start:end]],items_popularity[top_iids[start:end]],marker='D',color=color)
        # fig.savefig(os.path.join(self.DIRS['img'],"corr_popent_"+self.get_id()+".png"))

        # for idx_uid in tqdm(range(num_users)):
        #     uid = uids[idx_uid]
        #     self.results[uid].extend(top_iids)
        # self.save_results()
