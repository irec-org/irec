import numpy as np
from tqdm import tqdm
from .ExperimentalInteractor import ExperimentalInteractor
from . import Entropy, MostPopular
import matplotlib.pyplot as plt
import os
import scipy.stats
class MostRepresentative(ExperimentalInteractor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_representativeness(items_latent_factors):
        # return np.linalg.norm(items_latent_factors,axis=1,ord=np.inf)
        return np.sum(items_latent_factors**2,axis=1)

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[2],(self.train_dataset.data[0],self.train_dataset.data[1])),(self.train_dataset.users_num,self.train_dataset.items_num))
        # uids = self.test_users
        self.items_representativeness = self.get_items_representativeness(items_latent_factors)

    def predict(self,uid,candidate_items,num_req_items):
        items_score = self.items_representativeness[candidate_items]
        return items_score, None

        # top_iids = list(reversed(np.argsort(items_representativeness)))[:self.get_iterations()]

        # num_users = len(uids)
        # items_entropy = Entropy.get_items_entropy(self.train_consumption_matrix)
        # items_popularity = MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)

        # correlation = scipy.stats.pearsonr(items_entropy,items_popularity)[0]
        # fig, ax = plt.subplots()
        # ax.scatter(items_entropy,items_popularity,marker="D",color='darkblue')
        # ax.set_ylabel("Popularity")
        # ax.set_xlabel("Entropy")
        # ax.text(0.3, 0.9 , f'Correlation coefficient: {correlation:.2f}', color='k',
        #         ha='center', va='center',
        #         bbox=dict(facecolor='none', edgecolor='k', pad=10.0),
        #         transform = ax.transAxes)
        # for start, end, color in [(0,10,'green'),(10,20,'red'),(20,30,'darkred'),(30,40,'yellow'),(40,50,'orange')]:
        #     ax.scatter(items_entropy[top_iids[start:end]],items_popularity[top_iids[start:end]],marker='D',color=color)
        # fig.savefig(os.path.join(self.DIRS['img'],"corr_popent_"+self.get_name()+".png"))

        # for idx_uid in tqdm(range(num_users)):
        #     uid = uids[idx_uid]
        #     self.results[uid].extend(top_iids)
        # self.save_results()
