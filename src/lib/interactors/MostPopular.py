import numpy as np
from tqdm import tqdm
from .ExperimentalInteractor import ExperimentalInteractor
import matplotlib.pyplot as plt
import os
import scipy.sparse
class MostPopular(ExperimentalInteractor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_popularity(consumption_matrix, normalize=True):
        lowest_value = np.min(consumption_matrix)
        if not isinstance(consumption_matrix,scipy.sparse.spmatrix):
            items_popularity = np.count_nonzero(consumption_matrix>lowest_value,axis=0)
        else:
            items_popularity = np.array(np.sum(consumption_matrix>lowest_value,axis=0)).flatten()

        if normalize:
            items_popularity = items_popularity/consumption_matrix.shape[0]
                
        return items_popularity

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[:,2],(self.train_dataset.data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.users_num,self.train_dataset.items_num))
        # num_users = len(uids)
        # mask = np.ones(self.consumption_matrix.shape[0], dtype=bool)
        # mask[uids] = 0
        # num_train_users = np.count_nonzero(mask)

        self.items_popularity = self.get_items_popularity(self.train_consumption_matrix)

    def predict(self,uid,candidate_items,num_req_items):
        items_score = self.items_popularity[candidate_items]
        return items_score, None

    def update(self,uid,item,reward,additional_data):
        pass

        # fig, ax = plt.subplots()
        # ax.hist(items_popularity,color='k')
        # ax.set_xlabel("Popularity")
        # ax.set_ylabel("#Items")
        # fig.savefig(os.path.join(self.DIRS['img'],"popularity_"+self.get_id()+".png"))

        # top_iids = list(reversed(np.argsort(items_popularity)))[:self.get_iterations()]
        # uids = self.test_users
        # num_users = len(uids)
        # for idx_uid in tqdm(range(num_users)):
        #     uid = uids[idx_uid]
        #     self.results[uid].extend(top_iids)
        # self.save_results()
